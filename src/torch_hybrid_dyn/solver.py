from typing import TYPE_CHECKING

import torch
from torch import nn
from torchdiffeq import odeint, odeint_event

# from torchdiffeq import odeint_adjoint as odeint
from .dyn_mods import (
    NeuralDynamicsModule,
    NeuralEventModule,
    NeuralForceModule,
    NeuralResetModule,
)

if TYPE_CHECKING:
    from bidict import bidict  # noqa: TC004

    from .dyn_mods import (
        AnalyticalDynamicsModule,
        AnalyticalEventModule,
        AnalyticalForceModule,
        AnalyticalResetModule,
    )


class HybridStateObject(nn.Module):
    def __init__(
        self, n_hybrid_states: int, initial_state: (torch.Tensor | None) = None
    ):
        super().__init__()
        if initial_state is None:
            initial_state = torch.tensor([0])
        self.register_buffer("hybrid_state", initial_state)
        self.n_hybrid_states = n_hybrid_states

    def get_hybrid_state(self) -> torch.Tensor:
        return self.hybrid_state

    def set_hybrid_state(self, val: torch.Tensor):
        self.hybrid_state = torch.clamp(
            torch.round(val).int(), min=0, max=self.n_hybrid_states - 1
        )


class NNDynamicsObject(nn.Module):
    def __init__(
        self,
        force_module: (None | AnalyticalForceModule | NeuralForceModule) = None,
        event_module: (None | AnalyticalEventModule | NeuralEventModule) = None,
        reset_module: (None | AnalyticalResetModule | NeuralResetModule) = None,
        dynamics_module: (
            None | AnalyticalDynamicsModule | NeuralDynamicsModule
        ) = None,
        n_state_variables=1,
        n_constraints=1,
        hybrid_state_index_dict: (bidict | None) = None,
        solver_atol=1e-8,
        solver_rtol=1e-8,
        solver_options: (dict | None) = None,
        time_step=1e-7,
    ):
        if hybrid_state_index_dict is None:
            hybrid_state_index_dict = bidict()
        if solver_options is None:
            solver_options = {}
        super().__init__()
        self.n_state_variables = n_state_variables
        self.hybrid_state_index_dict = hybrid_state_index_dict
        self.n_hybrid_states = len(hybrid_state_index_dict)
        self.n_constraints = n_constraints
        self.solver_atol = solver_atol
        self.solver_rtol = solver_rtol
        self.solver_options = solver_options
        self.time_step = time_step

        self.hybrid_state_object = HybridStateObject(self.n_hybrid_states)

        self.event_times_failsafe = 100

        if force_module is None:
            self.force_module = NeuralForceModule(
                self.n_state_variables, self.get_hybrid_state
            )
        else:
            self.force_module = force_module
            self.force_module.set_hybrid_state_fn(
                self.hybrid_state_object.get_hybrid_state
            )

        if event_module is None:
            self.event_module = NeuralEventModule(
                self.force_module,
                self.n_state_variables,
                self.hybrid_state_object.get_hybrid_state,
                self.n_constraints,
                self.hybrid_state_index_dict,
            )
        else:
            self.event_module = event_module
            self.event_module.set_hybrid_state_fn(
                self.hybrid_state_object.get_hybrid_state
            )

        if reset_module is None:
            self.reset_module = NeuralResetModule(
                self.force_module,
                self.n_state_variables,
                self.hybrid_state_object.get_hybrid_state,
            )
            # uncomment to carry-forward the event function net
            # self.reset_module = NeuralResetModule(self.force_module,self.n_state_variables,self.get_hybrid_state,self.event_module)
        else:
            self.reset_module = reset_module
            self.reset_module.set_hybrid_state_fn(
                self.hybrid_state_object.get_hybrid_state
            )

        if dynamics_module is None:
            self.dynamics_module = NeuralDynamicsModule(
                self.force_module,
                self.event_module,
                self.n_state_variables,
                self.hybrid_state_object.get_hybrid_state,
            )
        else:
            self.dynamics_module = dynamics_module
            self.dynamics_module.set_hybrid_state_fn(
                self.hybrid_state_object.get_hybrid_state
            )

        self.batched_accel_fn = torch.vmap(dynamics_module)

    def simulate(
        self,
        times: torch.Tensor,
        return_accel=False,
        initial_state=None,
        local_state=False,
    ):
        device = times.device
        t0 = times[0:1]

        # Add a terminal time to the event function.
        def event_fn(t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            if t > (times[-1] + self.time_step):
                return torch.zeros([], device=device)
            event_fval = self.event_module(t, state)
            return event_fval

        if local_state:
            local_state_object = HybridStateObject(self.n_hybrid_states)
            self.force_module.set_hybrid_state_fn(local_state_object.get_hybrid_state)
            self.event_module.set_hybrid_state_fn(local_state_object.get_hybrid_state)
            self.reset_module.set_hybrid_state_fn(local_state_object.get_hybrid_state)
            self.dynamics_module.set_hybrid_state_fn(
                local_state_object.get_hybrid_state
            )
        else:
            local_state_object = self.hybrid_state_object

        # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
        # must appear in the state in the current implementation.
        if initial_state is not None:
            state = initial_state[:-1]
            local_state_object.set_hybrid_state(initial_state[-1:])
        else:
            state = torch.zeros(self.n_state_variables * 2, device=device)
            local_state_object.set_hybrid_state(torch.tensor([0], device=device))

        if torch.is_tensor(self.event_module.event_params):
            state = torch.cat(state, self.event_module.event_params)

        all_times = [t0]
        event_times: list[torch.Tensor] = []

        hybrid_states = [local_state_object.get_hybrid_state()]

        trajectory = [torch.unsqueeze(state[: self.n_state_variables * 2], 0)]
        if return_accel:
            acceleration = [
                self.dynamics_module(state)[
                    self.n_state_variables : self.n_state_variables * 2
                ]
            ]

        event_count = 0

        while t0 < times[-1]:
            # get event time
            event_t, solution = odeint_event(
                self.dynamics_module,
                state,
                t0,
                event_fn=event_fn,
                atol=self.solver_atol,
                rtol=self.solver_rtol,
                method="dopri5",
                options=self.solver_options,
            )
            if torch.any(torch.isnan(solution)):
                print("oh no")
                odeint_event(
                    self.dynamics_module,
                    state,
                    t0,
                    event_fn=event_fn,
                    atol=self.solver_atol,
                    rtol=self.solver_rtol,
                    method="dopri5",
                    options=self.solver_options,
                )

            event_count += 1

            # interval is the vector t0, all times <= event_t
            interval_ts = times[times > t0]
            # skip odeint if we start in an event function or exceeded our max number of events
            if event_t == t0 or event_count > self.event_times_failsafe:
                print("initial event error")
                print("Initial condition: " + str(initial_state))
                all_times.append(interval_ts)
                trajectory.append(state.expand(len(interval_ts), len(state)))
                break
            # normal ode if we finish the trajectory
            elif event_t > times[-1]:
                ode_ts = torch.cat((t0.reshape(-1), interval_ts.reshape(-1)))
                # odeint over the interval
                solution_ = odeint(
                    self.dynamics_module,
                    state,
                    ode_ts,
                    atol=self.solver_atol,
                    rtol=self.solver_rtol,
                    method="dopri5",
                    options=self.solver_options,
                )
                all_times.append(interval_ts)
                trajectory.append(solution_[1:, : self.n_state_variables * 2])
                if return_accel:
                    acceleration.append(
                        self.batched_accel_fn(solution_)[
                            self.n_state_variables : self.n_state_variables * 2
                        ]
                    )
                break
            # event based ode otherwise
            else:
                interval_ts = interval_ts[interval_ts < event_t]
                ode_ts = torch.cat(
                    (t0.reshape(-1), interval_ts.reshape(-1), event_t.reshape(-1))
                )
                # odeint over the interval
                solution_ = odeint(
                    self.dynamics_module,
                    state,
                    ode_ts,
                    atol=self.solver_atol,
                    rtol=self.solver_rtol,
                    method="dopri5",
                    options=self.solver_options,
                )

                state = solution[-1]

                # update velocity instantaneously.
                new_state = self.reset_module(event_t, state)
                local_state_object.set_hybrid_state(new_state[-1:])

                # advance the position a little bit to avoid re-triggering the event fn.
                # pos = new_state[0:self.n_state_variables]
                # vel = new_state[self.n_state_variables:-1]
                # adv = self.time_step*self.dynamics_module(event_t, new_state)[0:self.n_state_variables]

                # state = torch.cat((pos+adv,vel))
                state = new_state[:-1]
                t0 = event_t

                if len(interval_ts > 0):
                    all_times.append(interval_ts)
                    trajectory.append(solution_[1:-1, : self.n_state_variables * 2])
                    if return_accel:
                        acceleration.append(
                            self.batched_accel_fn(
                                solution_[1:-1, : self.n_state_variables * 2]
                            )[self.n_state_variables : self.n_state_variables * 2]
                        )
                    hybrid_states.append(local_state_object.get_hybrid_state())
                    event_times.append(event_t)
                else:
                    hybrid_states[-1] = local_state_object.get_hybrid_state()

        all_times = torch.cat(all_times)
        trajectory = torch.transpose(torch.cat(trajectory), 0, 1)
        if return_accel:
            acceleration = torch.transpose(torch.cat(acceleration), 0, 1)
            return all_times, trajectory, acceleration, hybrid_states, event_times
        else:
            return all_times, trajectory, hybrid_states, event_times
