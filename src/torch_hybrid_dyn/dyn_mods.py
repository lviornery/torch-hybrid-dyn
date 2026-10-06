from typing import TYPE_CHECKING

import torch
from torch import nn

from .net import MLP, SLLMLP, FunctionalMLP

if TYPE_CHECKING:
    from collections.abc import Callable

    from bidict import bidict

    from .analytical_prims import (
        ConsForce,
        ConsPos,
        Dynamics,
        ImpactComp,
        LiftoffComp,
        Reduction,
    )


class HybridDynamicsModule(nn.Module):
    def __init__(
        self, n_state_variables: int, hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None
    ):
        super().__init__()
        self.n_state_variables = n_state_variables
        self.hybrid_state_fn = hybrid_state_fn

    def set_hybrid_state_fn(self, hybrid_state_fn):
        self.hybrid_state_fn = hybrid_state_fn

    def get_dyn_state(self, state) -> torch.Tensor:
        return torch.cat(
            (state[0 : self.n_state_variables * 2], self.hybrid_state_fn())
        )


class AnalyticalForceModule(HybridDynamicsModule):
    def __init__(
        self,
        force_fn: Callable[[torch.Tensor],torch.Tensor],
        n_state_variables: int,
        hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None,
    ):
        super().__init__(n_state_variables, hybrid_state_fn)

        self.force_fn = force_fn

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        dyn_state = self.get_dyn_state(state)
        force = self.force_fn(torch.cat((torch.unsqueeze(t, -1),dyn_state)))
        return force


class NeuralForceModule(HybridDynamicsModule):
    def __init__(
        self,
        n_state_variables: int,
        hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None,
        use_time=True,
        use_position=True,
        use_velocity=True,
        use_hybrid_state=True,
        net_hidden_size=12,
        net_hidden_depth=2,
    ):
        super().__init__(n_state_variables, hybrid_state_fn)

        self.use_time = use_time
        self.use_position = use_position
        self.use_velocity = use_velocity
        self.use_hybrid_state = use_hybrid_state

        in_dim = sum(
            self.n_state_variables
            for use_option in [self.use_position, self.use_velocity]
            if use_option
        )
        in_dim += sum(
            1 for use_option in [self.use_time, self.use_hybrid_state] if use_option
        )

        self.force_net = MLP(
            in_dim, n_state_variables, net_hidden_size, net_hidden_depth
        )

    def getNetState(self, t: torch.Tensor, dyn_state: torch.Tensor) -> torch.Tensor:
        net_state = []
        if self.use_time:
            net_state.append(torch.unsqueeze(t, -1))
        if self.use_position:
            net_state.append(dyn_state[: self.n_state_variables])
        if self.use_velocity:
            net_state.append(
                dyn_state[self.n_state_variables : self.n_state_variables * 2]
            )
        if self.use_hybrid_state:
            net_state.append(dyn_state[-1:])
        if len(net_state) == 0:
            return torch.empty(0, device=t.device)
        else:
            return torch.cat(net_state)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        dyn_state = self.get_dyn_state(state)

        net_state = self.getNetState(t, dyn_state)

        force = self.force_net(net_state)
        return force


class AnalyticalEventModule(HybridDynamicsModule):
    def __init__(
        self,
        force_module: AnalyticalForceModule | NeuralForceModule,
        n_state_variables: int,
        cons_pos_module: ConsPos,
        cons_force_module: ConsForce,
        hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None,
        n_constraints=1,
        hybrid_state_index_dict: (bidict | None) = None,
    ):
        super().__init__(n_state_variables, hybrid_state_fn)

        self.out_dim = 0

        self.force_module = force_module

        self.cons_pos_module = cons_pos_module
        self.cons_force_module = cons_force_module

        self.event_params = []

        cons_masks = []

        if hybrid_state_index_dict:
            self.use_masks = True
            for index_dict_entry in hybrid_state_index_dict.values():
                impact_mask = torch.zeros(n_constraints).bool()
                force_mask = torch.ones(n_constraints).bool()
                for idx in index_dict_entry:
                    impact_mask[idx] = True
                    force_mask[idx] = False
                combined_mask = torch.cat((impact_mask, force_mask))
                cons_masks.append(combined_mask)
            self.register_buffer("cons_masks", torch.detach(torch.stack(cons_masks)))
        else:
            self.use_masks = False

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: event computation must use variables from the state.
        dyn_state = self.get_dyn_state(state)
        forces = self.force_module(t, state)

        # replace or augment with neural net output for ML
        cons_impacts = self.cons_pos_module(dyn_state)
        cons_forces = self.cons_force_module(forces, dyn_state)
        event = torch.cat((cons_impacts, cons_forces))

        # non-gradient-based masking for faster backprop
        if self.use_masks:
            hybrid_state = dyn_state[-1].int().item()
            cons_mask = self.cons_masks[hybrid_state]
            event.masked_fill_(cons_mask, 1.0)

        return event


class NeuralEventModule(HybridDynamicsModule):
    def __init__(
        self,
        force_module: AnalyticalForceModule | NeuralForceModule,
        n_state_variables: int,
        hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None,
        n_constraints=0,
        hybrid_state_index_dict: (bidict | None) = None,
        use_position=True,
        use_velocity=True,
        use_force=True,
        use_hybrid_state=True,
        net_hidden_size=12,
        net_hidden_depth=2,
    ):
        super().__init__(n_state_variables, hybrid_state_fn)

        self.out_dim = n_constraints * 2

        self.force_module = force_module

        self.use_position = use_position
        self.use_velocity = use_velocity
        self.use_force = use_force
        self.use_hybrid_state = use_hybrid_state

        in_dim = sum(
            self.n_state_variables
            for use_option in [self.use_position, self.use_velocity, self.use_force]
            if use_option
        )
        if self.use_hybrid_state:
            in_dim += 1

        self.event_net = FunctionalMLP(
            in_dim, self.out_dim, net_hidden_size, net_hidden_depth
        )
        self.event_params = nn.Parameter(self.event_net.export_init_params())

        cons_masks = []

        if hybrid_state_index_dict:
            for index_dict_entry in hybrid_state_index_dict.values():
                impact_mask = torch.zeros(n_constraints).bool()
                force_mask = torch.ones(n_constraints).bool()
                for idx in index_dict_entry:
                    impact_mask[idx] = True
                    force_mask[idx] = False
                combined_mask = torch.cat((impact_mask, force_mask))
                cons_masks.append(combined_mask)
            self.register_buffer("cons_masks", torch.detach(torch.stack(cons_masks)))

    def getNetState(self, dyn_state: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        net_state = []
        if self.use_position:
            net_state.append(dyn_state[: self.n_state_variables])
        if self.use_velocity:
            net_state.append(
                dyn_state[self.n_state_variables : self.n_state_variables * 2]
            )
        if self.use_force:
            net_state.append(force)
        if self.use_hybrid_state:
            net_state.append(dyn_state[-1:])
        if len(net_state) == 0:
            return torch.empty(0, device=dyn_state.device)
        else:
            return torch.cat(net_state)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: event computation must use variables from the state.
        dyn_state = self.get_dyn_state(state)

        force = self.force_module(t, state)
        net_state = self.getNetState(dyn_state, force)

        event = self.event_net(net_state, state[self.n_state_variables * 2 :])

        if self.cons_masks:
            # non-gradient-based masking for faster backprop
            hybrid_state = dyn_state[-1].int().item()
            cons_mask = self.cons_masks[hybrid_state]
            event.masked_fill_(cons_mask, 1.0)

        return torch.tanh(event)


class AnalyticalResetModule(HybridDynamicsModule):
    def __init__(
        self,
        force_module: AnalyticalForceModule | NeuralForceModule,
        n_state_variables: int,
        impact_comp_fn: ImpactComp,
        liftoff_comp_fn: LiftoffComp,
        hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None,
        event_module: None = None,
    ):
        super().__init__(n_state_variables, hybrid_state_fn)
        self.force_module = force_module

        self.impact_comp_fn = impact_comp_fn
        self.liftoff_comp_fn = liftoff_comp_fn

    def forward(self, t: torch.Tensor, state: torch.Tensor):
        dyn_state = self.get_dyn_state(state)
        forces = self.force_module(t, state)

        # replace or augment with neural net output
        new_deriv_state = self.impact_comp_fn(dyn_state)
        new_state = torch.cat((state[0 : self.n_state_variables], new_deriv_state))
        new_state[self.n_state_variables * 2] = self.liftoff_comp_fn(forces, new_state)

        return new_state


class NeuralResetModule(HybridDynamicsModule):
    def __init__(
        self,
        force_module: AnalyticalForceModule | NeuralForceModule,
        n_state_variables: int,
        hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None,
        event_module: AnalyticalEventModule | NeuralEventModule | None = None,
        use_position=True,
        use_velocity=True,
        use_force=True,
        use_hybrid_state=True,
        use_event=True,
        net_hidden_size=12,
        net_hidden_depth=2,
    ):
        super().__init__(n_state_variables, hybrid_state_fn)

        self.force_module = force_module
        self.event_module = event_module

        self.use_position = use_position
        self.use_velocity = use_velocity
        self.use_force = use_force
        self.use_hybrid_state = use_hybrid_state
        self.use_event = self.event_module and use_event

        in_dim = sum(
            self.n_state_variables
            for use_option in [self.use_position, self.use_velocity, self.use_force]
            if use_option
        )
        if self.use_event:
            in_dim += self.event_module.out_dim
        if self.use_hybrid_state:
            in_dim += 1

        self.reset_net = MLP(
            in_dim, n_state_variables + 1, net_hidden_size, net_hidden_depth
        )

    def getNetState(self, dyn_state: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        net_state = []
        if self.use_position:
            net_state.append(dyn_state[: self.n_state_variables])
        if self.use_velocity:
            net_state.append(
                dyn_state[self.n_state_variables : self.n_state_variables * 2]
            )
        if self.use_force:
            net_state.append(force)
        if self.use_hybrid_state:
            net_state.append(dyn_state[-1:])
        if len(net_state) == 0:
            return torch.empty(0, device=dyn_state.device)
        else:
            return torch.cat(net_state)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        dyn_state = self.get_dyn_state(state)

        force = self.force_module(t, state)
        net_state = self.getNetState(dyn_state, force)

        if self.use_event:
            event_net_state = self.event_module.getNetState(dyn_state, force)
            event = self.event_module.event_net(event_net_state)
            net_state = torch.cat((net_state, event))

        reset = self.reset_net(net_state)

        return torch.cat((state[0 : self.n_state_variables], reset))


class AnalyticalDynamicsModule(HybridDynamicsModule):
    def __init__(
        self,
        force_module: AnalyticalForceModule | NeuralForceModule,
        event_module: AnalyticalEventModule | NeuralEventModule,
        n_state_variables: int,
        dynamics_fns: Dynamics,
        hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None,
        reduction_modules: None = None,
    ):
        super().__init__(n_state_variables, hybrid_state_fn)

        self.force_module = force_module

        self.use_reductions = False
        self.dynamics_fn = dynamics_fns

        if torch.is_tensor(event_module.event_params):
            self.register_buffer(
                "rest_zeros", torch.zeros_like(event_module.event_params)
            )
        else:
            self.register_buffer("rest_zeros", torch.empty(0))

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        dyn_state = self.get_dyn_state(state)
        force = self.force_module(t, state)

        # replace or augment with neural net output
        accel = self.dynamics_fn(force, dyn_state)

        return torch.cat(
            (
                state[self.n_state_variables : self.n_state_variables * 2],
                accel,
                self.rest_zeros,
            )
        )


class NeuralDynamicsModule(HybridDynamicsModule):
    def __init__(
        self,
        force_module: AnalyticalForceModule | NeuralForceModule,
        event_module: AnalyticalEventModule | NeuralEventModule,
        n_state_variables: int,
        hybrid_state_fn: (Callable[[],torch.Tensor] | None) = None,
        use_sll=False,
        reduction_modules: (None | list[None | Reduction]) = None,
        use_position=True,
        use_velocity=True,
        use_force=True,
        use_hybrid_state=True,
        net_hidden_size=12,
        net_hidden_depth=8,
        net_divergence_depth=-4,
    ):
        super().__init__(n_state_variables, hybrid_state_fn)

        self.force_module = force_module

        self.use_position = use_position
        self.use_velocity = use_velocity
        self.use_force = use_force
        self.use_hybrid_state = use_hybrid_state

        in_dim = sum(
            self.n_state_variables
            for use_option in [self.use_position, self.use_velocity, self.use_force]
            if use_option
        )

        if reduction_modules is not None:
            self.use_reductions = True
            self.use_sll = use_sll
            self.reduction_modules = nn.ModuleList(reduction_modules)
            self.reduced_dims = []
            for reduction_module in reduction_modules:
                if reduction_module is None:
                    self.reduced_dims.append(n_state_variables)
                elif reduction_module.no_output:
                    self.reduced_dims.append(0)
                else:
                    self.reduced_dims.append(
                        n_state_variables - reduction_module.n_reductions
                    )
            if self.use_sll:
                self.dynamics_net = SLLMLP(
                    in_dim,
                    self.reduced_dims,
                    net_hidden_size,
                    net_hidden_depth,
                    divergence_depth=net_divergence_depth,
                )
            else:
                dynamics_nets = []
                for reduced_n_state_variables in self.reduced_dims:
                    if reduced_n_state_variables == 0:
                        dynamics_nets.append(None)
                    else:
                        dynamics_nets.append(
                            MLP(
                                in_dim,
                                reduced_n_state_variables,
                                net_hidden_size,
                                net_hidden_depth,
                            )
                        )
                self.dynamics_net = nn.ModuleList(dynamics_nets)
        else:
            if self.use_hybrid_state:
                in_dim += 1
            self.use_reductions = False
            self.dynamics_net = MLP(
                in_dim, n_state_variables, net_hidden_size, net_hidden_depth
            )

        if torch.is_tensor(event_module.event_params):
            self.register_buffer(
                "rest_zeros", torch.zeros_like(event_module.event_params)
            )
        else:
            self.register_buffer("rest_zeros", torch.empty(0))

    def getNetState(
        self,
        dyn_state: torch.Tensor,
        force: torch.Tensor,
        H_matrix: (None | torch.Tensor) = None,
    ) -> torch.Tensor:
        net_state = []
        if self.use_position:
            net_state.append(dyn_state[: self.n_state_variables])
        if self.use_velocity:
            net_state.append(
                dyn_state[self.n_state_variables : self.n_state_variables * 2]
            )
        if self.use_force:
            if H_matrix is not None:
                y_force = torch.matmul(torch.transpose(H_matrix, 0, 1), force)
            net_state.append(y_force)
        if self.use_hybrid_state and not self.use_reductions:
            net_state.append(dyn_state[-1:])
        if len(net_state) == 0:
            return torch.empty(0, device=dyn_state.device)
        else:
            return torch.cat(net_state)

    def forward(self, t:torch.Tensor, state:torch.Tensor) -> torch.Tensor:
        dyn_state = self.get_dyn_state(state)
        force = self.force_module(t, state)

        state_index = dyn_state[-1].int().item()
        if self.use_reductions:
            reduced_dim = self.reduced_dims[state_index]
            if reduced_dim == 0:
                accel = torch.zeros(self.n_state_variables, device=t.device)
            elif self.reduction_modules[state_index] is None:
                net_state = self.getNetState(dyn_state, force)
                if self.use_sll:
                    accel = self.dynamics_net(net_state, state_index)
                else:
                    accel = self.dynamics_net[state_index](net_state)
            else:
                (y_state, H_matrix, H_dot_matrix) = self.reduction_modules[state_index](
                    dyn_state[:-1]
                )
                if self.use_sll:
                    net_state = self.getNetState(dyn_state, force)
                    y_accel = self.dynamics_net(net_state, state_index)
                else:
                    y_net_state = self.getNetState(y_state, force, H_matrix=H_matrix)
                    y_accel = self.dynamics_net[state_index](y_net_state)

                accel = torch.matmul(H_matrix, y_accel) + torch.matmul(
                    H_dot_matrix, y_state[reduced_dim:]
                )
        else:
            net_state = self.getNetState(dyn_state, force)
            accel = self.dynamics_net(net_state)

        if torch.any(torch.isnan(accel)):
            print("nan encountered with dyn_state:" + str(dyn_state))
            print("net params:")
            if self.use_reductions and not self.use_sll:
                print([p for p in self.dynamics_net[state_index].parameters()])
            else:
                print([p for p in self.dynamics_net.parameters()])

        return torch.cat(
            (
                state[self.n_state_variables : self.n_state_variables * 2],
                accel,
                self.rest_zeros,
            )
        )
