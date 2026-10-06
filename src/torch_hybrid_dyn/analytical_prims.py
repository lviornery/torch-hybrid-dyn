import itertools
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable  # noqa: TC004

    import bidict  # noqa: TC004

torch.set_default_dtype(torch.float64)


class MBar(nn.Module):
    def __init__(
        self, entries: list[list[Callable[[torch.Tensor], torch.Tensor] | torch.Tensor]]
    ):
        super().__init__()

        self.dim = len(entries)
        entryList = list(itertools.chain.from_iterable(entries))

        if all(torch.is_tensor(entry) for entry in entryList):
            self.is_static = True
            self.register_buffer(
                "static_mbar", torch.reshape(torch.stack(entryList), (self.dim, -1))
            )
        else:
            self.mbar_entries: list[
                Callable[[torch.Tensor], torch.Tensor] | torch.Tensor
            ] = []
            self.is_static = False
            for idx, entry in enumerate(entryList):
                if torch.is_tensor(entry):
                    self.register_buffer("mbar_" + str(idx), entry)
                    self.mbar_entries.append(self.get_buffer("mbar_" + str(idx)))
                else:
                    self.mbar_entries.append(entry)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.is_static:
            return self.static_mbar
        else:
            return torch.reshape(
                torch.stack(
                    [
                        entry if torch.is_tensor(entry) else entry(state)
                        for entry in self.mbar_entries
                    ]
                ),
                (self.dim, -1),
            )


class ARow(nn.Module):
    def __init__(
        self, A_constraint: (Callable[[torch.Tensor], torch.Tensor] | torch.Tensor)
    ):
        super().__init__()

        if torch.is_tensor(A_constraint):
            self.is_static = True
            self.register_buffer("static_A", A_constraint)
        else:
            self.is_static = False
            self.A_constraint = A_constraint

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.is_static:
            return self.static_A
        else:
            return self.A_constraint(state)


class A(nn.Module):
    def __init__(self, A_rows: list[ARow]):
        super().__init__()

        if all(A_row.is_static for A_row in A_rows):
            self.is_static = True
            self.register_buffer(
                "static_A", torch.stack([A_row.static_A for A_row in A_rows])
            )
        else:
            self.is_static = False
            self.A_rows = nn.ModuleList(A_rows)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.is_static:
            return self.static_A
        else:
            return torch.stack([A_row(state) for A_row in self.A_rows])


class AT(nn.Module):
    def __init__(self, A_constraint: A):
        super().__init__()

        self.A_constraint = A_constraint

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.transpose(self.A_constraint(state), 0, 1)


class Mdag(nn.Module):
    def __init__(self, mbar: MBar, state_A: A, state_A_T: AT):
        super().__init__()

        self.mbar = mbar
        self.state_A = state_A
        self.state_A_T = state_A_T

        self.register_buffer("eye", torch.eye(self.mbar.dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # mdag = inv(mbar)*(eye-AT*inv(A*inv(mbar)*AT)*A*inv(mbar))
        # mdag = inv(mbar)*(eye-W*X)
        # W = AT*inv(Y)
        # X = A*inv(mbar)
        # Y = A*Z
        # Z = inv(M)*AT
        Z = torch.linalg.solve(self.mbar(state), self.state_A_T(state))
        Y = torch.matmul(self.state_A(state), Z)
        X = torch.linalg.solve(self.mbar(state), self.state_A(state), left=False)
        W = torch.linalg.solve(Y, self.state_A_T(state), left=False)
        ret = torch.linalg.solve(self.mbar(state), self.eye - torch.matmul(W, X))
        return ret


class AdagT(nn.Module):
    def __init__(self, mbar: MBar, state_A: A, state_A_T: AT):
        super().__init__()

        self.mbar = mbar
        self.state_A = state_A
        self.state_A_T = state_A_T

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # adag = inv(mbar)*AT*inv(A*inv(mbar)*AT)
        # adag = inv(mbar)*X
        # X = AT*inv(Y)
        # Y = A*Z
        # Z = inv(M)*AT
        Z = torch.linalg.solve(self.mbar(state), self.state_A_T(state))
        Y = torch.matmul(self.state_A(state), Z)
        X = torch.linalg.solve(Y, self.state_A_T(state), left=False)
        ret = torch.linalg.solve(self.mbar(state), X)
        return ret


class Adag(nn.Module):
    def __init__(self, state_Adag_T: AdagT):
        super().__init__()

        self.state_Adag_T = state_Adag_T

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.transpose(self.state_Adag_T(state), 0, 1)


class Lambda(nn.Module):
    def __init__(self, mbar: MBar, state_A: A, state_A_T: AT):
        super().__init__()

        self.mbar = mbar
        self.state_A = state_A
        self.state_A_T = state_A_T

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # lambda = -inv(A*inv(mbar)*AT)
        # lambda = -inv(Y)
        # Y = A*Z
        # Z = inv(M)*AT
        Z = torch.linalg.solve(self.mbar(state), self.state_A_T(state))
        Y = torch.matmul(self.state_A(state), Z)
        ret = -1 * torch.linalg.inv(Y)
        return ret


class DerivativeReduction(nn.Module):
    def __init__(
        self,
        coordinate_transform: None | Callable[[torch.Tensor], torch.Tensor] = None,
        Y_map: (None | Callable[[torch.Tensor], torch.Tensor] | torch.Tensor) = None,
        Y_dot_map: (
            None | Callable[[torch.Tensor], torch.Tensor] | torch.Tensor
        ) = None,
    ):
        super().__init__()

        if coordinate_transform is not None:
            self.Y_is_static = False
            self.Y_dot_is_static = False
            self.calculate_Y_map = True
            self.calculate_Y_dot_map = True
            self.coordinate_transform = coordinate_transform

        if torch.is_tensor(Y_map):
            self.Y_is_static = True
            self.register_buffer("static_Y", Y_map)
            self.Y_dot_is_static = True
            self.register_buffer("static_Y_dot", torch.zeros_like(Y_map))
        elif Y_map is not None:
            self.Y_is_static = False
            self.calculate_Y_map = False
            self.calculate_Y_dot_map = True
            self.Y_map = Y_map

        if torch.is_tensor(Y_dot_map):
            self.Y_dot_is_static = True
            self.register_buffer("static_Y_dot", Y_dot_map)
        elif Y_dot_map is not None:
            self.Y_dot_is_static = False
            self.calculate_Y_dot_map = False
            self.Y_dot_map = Y_dot_map

    def forward(
        self, state: torch.Tensor, get_Y_dot=True
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        n_state_variables = len(state) // 2
        Y_val = None
        if self.calculate_Y_map:
            Y_map = torch.func.jacrev(self.coordinate_transform)
        elif not self.Y_is_static:
            Y_val = self.static_Y
        else:
            Y_map = self.Y_map

        if self.calculate_Y_dot_map and get_Y_dot:
            return torch.autograd.functional.jvp(
                Y_map,
                state[:n_state_variables],
                state[n_state_variables:],
                create_graph=True,
            )

        if Y_val is None:
            Y_val = Y_map(state)

        if not get_Y_dot:
            return Y_val

        Y_dot_val = None
        if self.Y_dot_is_static:
            Y_dot_val = self.static_Y_dot
        else:
            Y_dot_val = self.Y_dot_map(state)

        return (Y_val, Y_dot_val)


class Reduction(nn.Module):
    def __init__(
        self,
        A_rows: list[ARow],
        A_dot_rows: list[ARow],
        coordinate_reductions: list[Callable[[torch.Tensor], torch.Tensor]],
        derivative_reductions: list[DerivativeReduction],
    ):
        super().__init__()

        if (
            (len(A_rows) == 0)
            or (len(A_dot_rows) == 0)
            or any(reduction is None for reduction in coordinate_reductions)
            or any(reduction is None for reduction in derivative_reductions)
        ):
            self.no_output = True
            self.register_buffer("null_coordinates", torch.tensor([]))
        else:
            self.no_output = False
            self.coord_reductions = coordinate_reductions
            self.n_reductions = len(coordinate_reductions)
            zero = torch.zeros(len(A_rows), self.n_reductions)
            id = torch.eye(self.n_reductions)
            rhs = torch.cat((zero, id))

            if all(A_row.is_static for A_row in A_rows) and all(
                Y_red.Y_is_static for Y_red in derivative_reductions
            ):
                self.deriv_is_static = True
                self.double_deriv_is_static = True
                Y_list = [Y_red.static_Y for Y_red in derivative_reductions]
                self.register_buffer("static_Y", torch.stack(Y_list))
                invm = torch.stack([A_row.static_A for A_row in A_rows] + Y_list)
                self.register_buffer("static_H", torch.linalg.solve(invm, rhs))
                self.double_deriv_is_static = True
                self.register_buffer("static_H_dot", torch.zeros_like(self.static_H))
            else:
                self.deriv_is_static = False
                self.A_rows = nn.ModuleList(A_rows)
                if all(A_dot_row.is_static for A_dot_row in A_dot_rows) and all(
                    Y_red.Y_dot_is_static for Y_red in derivative_reductions
                ):
                    self.double_deriv_is_static = True
                    # we don't need static_Y_dot for anything besides static_H_dot so we just don't register it
                    # H_dot = -1*inv(invm)*[A_dot Y_dot]T*inv(invm)*rhs
                    # H_dot = Y*Z
                    # Y = inv(invm)*[A_dot Y_dot]T
                    # Z = inv(invm)*rhs
                    ay_dot = torch.stack(
                        [A_dot_row.static_A for A_dot_row in A_dot_rows]
                        + [Y_red.static_Y_dot for Y_red in derivative_reductions]
                    )
                    H_dot_Y = torch.linalg.solve(invm, ay_dot)
                    H_dot_Z = torch.linalg.solve(invm, rhs)
                    self.register_buffer(
                        "static_H_dot", -1 * torch.matmul(H_dot_Y, H_dot_Z)
                    )
                else:
                    self.double_deriv_is_static = False
                    self.A_dot_rows = nn.ModuleList(A_dot_rows)

            if (not self.deriv_is_static) or (not self.double_deriv_is_static):
                self.register_buffer("static_rhs", rhs)
                self.derivative_reductions = nn.ModuleList(derivative_reductions)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.no_output:
            return (self.null_coordinates, self.null_coordinates, self.null_coordinates)
        else:
            n_state_variables = len(state) // 2
            y_state = None
            H_matrix = None
            H_dot_matrix = None
            y = torch.stack([coord_fn(state) for coord_fn in self.coord_reductions])
            if self.deriv_is_static:
                y_dot = torch.mv(self.static_Y, state[n_state_variables:])
                H_matrix = self.static_H
                H_dot_matrix = self.static_H_dot
            else:
                Y_dots = []
                Ys = []
                for Y_red in self.derivative_reductions:
                    (Y_matrix, Y_dot_matrix) = Y_red(state)
                    Y_dots.append(Y_dot_matrix)
                    Ys.append(Y_matrix)
                Y_matrix = torch.stack(Ys)
                Y_dot_matrix = torch.stack(Y_dots)
                y_dot = torch.mv(Y_matrix, state[n_state_variables:])
                invm = torch.stack(
                    [A_row(state[:n_state_variables]) for A_row in self.A_rows] + Ys
                )
                H_matrix = torch.linalg.solve(invm, self.static_rhs)
                if self.double_deriv_is_static:
                    H_dot_matrix = self.static_H_dot
                else:
                    a_dot = torch.stack(
                        [A_dot_row(state) for A_dot_row in self.A_dot_rows]
                    )
                    ay_dot = torch.cat((a_dot, Y_dot_matrix))
                    H_dot_Y = torch.linalg.solve(invm, ay_dot)
                    H_dot_Z = torch.linalg.solve(invm, self.static_rhs)
                    H_dot_matrix = -1 * torch.matmul(H_dot_Y, H_dot_Z)
            y_state = torch.cat((y, y_dot))
            return y_state, H_matrix, H_dot_matrix


class Dynamics(nn.Module):
    def __init__(
        self,
        mbar: MBar,
        rhs: type[nn.Module],
        state_mdags: list[None | Mdag],
        state_Adag_Ts: list[None | AdagT],
        state_A_dots: list[None | A],
    ):
        super().__init__()

        self.mbar = mbar
        self.rhs = rhs
        self.state_mdags = nn.ModuleList(state_mdags)
        self.state_Adag_Ts = nn.ModuleList(state_Adag_Ts)
        self.state_A_dots = nn.ModuleList(state_A_dots)

        self.n_state_variables = self.mbar.dim

    def forward(self, forces: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        eq_state = state[:-1]
        derivative_tensor = state[self.n_state_variables : self.n_state_variables * 2]
        hybrid_state = state[-1].int().item()

        if hybrid_state == 0:
            mbar = self.mbar(eq_state)
            rhs = self.rhs(forces, state)
            return torch.linalg.solve(mbar, rhs)
        else:
            m_dag = self.state_mdags[hybrid_state](eq_state)
            A_dag_T = self.state_Adag_Ts[hybrid_state](eq_state)
            A_dot = self.state_A_dots[hybrid_state](eq_state)
            rhs = self.rhs(forces, state)
            ret = torch.mv(m_dag, rhs) - torch.mv(
                torch.matmul(A_dag_T, A_dot), derivative_tensor
            )
            return ret


class ConsForce(nn.Module):
    def __init__(
        self,
        rhs: type[nn.Module],
        state_Adags: list[None | Adag],
        state_Lambdas: list[None | Lambda],
        state_A_dots: list[None | A],
        n_constraints: int,
        n_state_variables: int,
        state_index_dict: bidict,
        cons_thresh: float,
    ):
        super().__init__()

        self.rhs = rhs
        self.state_Adags = nn.ModuleList(state_Adags)
        self.state_Lambdas = nn.ModuleList(state_Lambdas)
        self.state_A_dots = nn.ModuleList(state_A_dots)

        self.n_state_variables = n_state_variables
        self.state_index_dict = state_index_dict
        self.cons_thresh = cons_thresh
        self.n_constraints = n_constraints

    def forward(self, forces: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        eq_state = state[:-1]
        derivative_tensor = state[self.n_state_variables : self.n_state_variables * 2]
        hybrid_state = state[-1].int().item()

        cons_force_tensor = torch.zeros(self.n_constraints, device=state.device)

        if hybrid_state != 0:
            A_dag = self.state_Adags[hybrid_state](eq_state)
            Lambda = self.state_Lambdas[hybrid_state](eq_state)
            A_dot = self.state_A_dots[hybrid_state](eq_state)
            state_index = self.state_index_dict[hybrid_state]
            rhs = self.rhs(forces, state)
            active_cons_forces = torch.mv(A_dag, rhs) - torch.mv(
                torch.matmul(Lambda, A_dot), derivative_tensor
            )
            for cons_force_index, cons_force in zip(state_index, active_cons_forces):
                cons_force_tensor[cons_force_index] = cons_force
        return cons_force_tensor


class ConsPos(nn.Module):
    def __init__(
        self, position_constraints: list[Callable[[torch.Tensor], torch.Tensor]]
    ):
        super().__init__()

        self.position_constraints = position_constraints

        self.range_constraints = range(len(position_constraints))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        eq_state = state[:-1]
        return torch.stack(
            [self.position_constraints[idx](eq_state) for idx in self.range_constraints]
        )


class ImpactComp(nn.Module):
    def __init__(
        self,
        mbar: MBar,
        state_mdags: list[None | Mdag],
        state_Adags: list[None | Adag],
        position_constraints: list[Callable[[torch.Tensor], torch.Tensor]],
        A_constraints: list[ARow],
        state_index_dict: bidict,
        con_thresh: float,
    ):
        super().__init__()

        self.mbar = mbar
        self.state_mdags = nn.ModuleList(state_mdags)
        self.state_Adags = nn.ModuleList(state_Adags)
        self.A_constraints = nn.ModuleList(A_constraints)
        self.position_constraints = position_constraints

        self.con_thresh = con_thresh
        self.state_index_dict = state_index_dict
        self.n_state_variables = self.mbar.dim
        self.range_constraints = range(len(position_constraints))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        eq_state = state[:-1]
        derivative_tensor = state[self.n_state_variables : self.n_state_variables * 2]
        new_state = state[self.n_state_variables :]

        mbar = self.mbar(eq_state)

        # set of contact constraints currently active
        contacts = {
            cons_idx
            for cons_idx, position_constraint in zip(
                self.range_constraints, self.position_constraints
            )
            if position_constraint(eq_state) is not None
            and torch.abs(position_constraint(eq_state)) < self.con_thresh
        }

        for hybrid_state, hybrid_state_contacts in self.state_index_dict.items():
            hybrid_state_contact_set = set(hybrid_state_contacts)
            # if, for EVERY hybrid state constraint in the state we're looking at,
            # that constraint is fulfilled, that state is a valid candidate state
            if hybrid_state != 0 and hybrid_state_contact_set.issubset(contacts):
                # check validity of impulses into this state
                Adag = self.state_Adags[hybrid_state](eq_state)
                if all(torch.mv(torch.matmul(Adag, mbar), derivative_tensor) <= 0):
                    mdag = self.state_mdags[hybrid_state](eq_state)
                    new_vel = torch.mv(torch.matmul(mdag, mbar), derivative_tensor)
                    extra_As = [
                        self.A_constraints[idx](eq_state)
                        for idx in contacts - hybrid_state_contact_set
                    ]
                    if all(torch.dot(extra_A, new_vel) > 0 for extra_A in extra_As):
                        new_state = torch.cat(
                            (new_vel, torch.tensor([hybrid_state], device=state.device))
                        )
                        break

        return new_state


class LiftoffComp(nn.Module):
    def __init__(
        self,
        dynamics: Dynamics,
        rhs: type[nn.Module],
        state_As: list[None | A],
        state_Adags: list[None | Adag],
        state_Lambdas: list[None | Lambda],
        state_A_dots: list[None | A],
        n_state_variables: int,
        state_index_dict: bidict,
        con_thresh: float,
    ):
        super().__init__()

        self.dynamics = dynamics

        self.rhs = rhs
        self.state_As = nn.ModuleList(state_As)
        self.state_Adags = nn.ModuleList(state_Adags)
        self.state_Lambdas = nn.ModuleList(state_Lambdas)
        self.state_A_dots = nn.ModuleList(state_A_dots)

        self.n_state_variables = n_state_variables
        self.state_index_dict = state_index_dict
        self.con_thresh = con_thresh
        self.range_hybrid_states = range(len(state_index_dict))

    def forward(self, forces: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        eq_state = state[:-1]
        derivative_tensor = state[self.n_state_variables : self.n_state_variables * 2]
        init_hybrid_state = state[-1].int().item()
        new_state = state[-1:]

        # get nominal constraint forces
        if init_hybrid_state != 0:
            state_cons_indices = self.state_index_dict[init_hybrid_state]
            state_A = self.state_As[init_hybrid_state](eq_state)
            state_A_dot = self.state_A_dots[init_hybrid_state](eq_state)
            state_A_dag = self.state_Adags[init_hybrid_state](eq_state)
            state_Lambda = self.state_Lambdas[init_hybrid_state](eq_state)
            state_rhs = self.rhs(forces, state)

            cons_forces = torch.mv(state_A_dag, state_rhs) - torch.mv(
                torch.matmul(state_Lambda, state_A_dot), derivative_tensor
            )

            cons_forces_large_neg_mask = cons_forces.lt(-self.con_thresh)

            # if every single force is large and negative, just return right away
            if torch.all(cons_forces_large_neg_mask):
                return new_state
            # if the only forces are large and negative or large and positive, identify the state based on the negative forces and return that
            cons_forces_large_pos_mask = torch.gt(cons_forces, self.con_thresh)
            if torch.all(
                torch.bitwise_or(cons_forces_large_neg_mask, cons_forces_large_pos_mask)
            ):
                new_cons = tuple(
                    cons_index
                    for force_index, cons_index in enumerate(state_cons_indices)
                    if cons_forces_large_neg_mask[force_index].item()
                )
                new_state = self.state_index_dict.inverse[new_cons]
                return new_state

            cons_forces_indeterminate_mask = torch.bitwise_and(
                torch.lt(cons_forces, self.con_thresh),
                torch.gt(cons_forces, -self.con_thresh),
            )

            large_neg_cons_force_indices = (
                cons_index
                for force_index, cons_index in enumerate(state_cons_indices)
                if cons_forces_large_neg_mask[force_index].item()
            )

            indeterminate_cons_force_indices = (
                cons_index
                for force_index, cons_index in enumerate(state_cons_indices)
                if cons_forces_indeterminate_mask[force_index].item()
            )

            active_cons_set = set(large_neg_cons_force_indices)
            potential_cons_set = set(indeterminate_cons_force_indices).union(
                active_cons_set
            )

            # precalculate since this doesn't change at all
            state_adot_part = torch.mv(state_A_dot, derivative_tensor)

            # reverse so we prefer maximizing the number of active forces
            for hybrid_state, hybrid_state_contacts in reversed(
                self.state_index_dict.items()
            ):
                new_state_contact_set = set(hybrid_state_contacts)
                if active_cons_set.issubset(
                    new_state_contact_set
                ) and potential_cons_set.issuperset(new_state_contact_set):
                    candidate_new_state = torch.tensor(
                        [hybrid_state], device=state.device
                    )
                    new_state_vec = torch.cat((eq_state, candidate_new_state))
                    new_state_accel = self.dynamics(forces, new_state_vec)
                    # if we're considering the free-flying state, make sure we're moving away from all of the previous state's constraints
                    if candidate_new_state == 0 and torch.all(
                        (torch.mv(state_A, new_state_accel) + state_adot_part).gt(0)
                    ):
                        new_state = candidate_new_state
                    else:
                        new_state_rhs = self.rhs(forces, new_state_vec)

                        new_state_cons_indices = self.state_index_dict[hybrid_state]
                        new_state_Adag = self.state_Adags[hybrid_state](eq_state)
                        new_state_Lambda = self.state_Lambdas[hybrid_state](eq_state)
                        new_state_A_dot = self.state_A_dots[hybrid_state](eq_state)

                        # for each of the original state's constraints, we should be EITHER moving away from them, or we should be maintaining them
                        # check constraints from the old state that we're moving away from
                        old_state_accel_ok_mask = (
                            torch.mv(state_A, new_state_accel) + state_adot_part
                        ).gt(0)
                        old_state_accel_ok_indices = (
                            cons_index
                            for force_index, cons_index in enumerate(state_cons_indices)
                            if old_state_accel_ok_mask[force_index]
                        )
                        # check constraints from the new state that we're maintaining
                        new_state_force_ok_mask = (
                            torch.mv(new_state_Adag, new_state_rhs)
                            - torch.mv(
                                torch.matmul(new_state_Lambda, new_state_A_dot),
                                derivative_tensor,
                            )
                        ).le(0)
                        new_state_force_ok_indices = (
                            cons_index
                            for force_index, cons_index in enumerate(
                                new_state_cons_indices
                            )
                            if new_state_force_ok_mask[force_index]
                        )

                        # if we're maintaining or moving away from every potentially active constraint from the old state, this is a valid new state
                        if (
                            set(
                                itertools.chain(
                                    old_state_accel_ok_indices,
                                    new_state_force_ok_indices,
                                )
                            )
                            == potential_cons_set
                        ):
                            new_state = candidate_new_state
                            break
        return new_state
