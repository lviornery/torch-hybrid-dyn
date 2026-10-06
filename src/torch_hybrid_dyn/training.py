import collections
import math
from typing import TYPE_CHECKING

import torch

from .util import NetType, SeriesType

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.synchronize import Lock

    from .data import MultiThreadDataloader, SingleThreadDataloader
    from .solver import NNDynamicsObject


class NetLearnRateMultipliers(collections.UserDict):
    def __init__(
        self,
        force_multiplier: float,
        event_multiplier: float,
        reset_multiplier: float,
        dynamics_multiplier: float,
    ):
        mapping = {
            NetType.FORCE: force_multiplier,
            NetType.EVENT: event_multiplier,
            NetType.RESET: reset_multiplier,
            NetType.DYNAMICS: dynamics_multiplier,
        }
        super().__init__(mapping)


class SeriesLearnRateMultipliers(collections.UserDict):
    def __init__(
        self,
        full_multipliers: NetLearnRateMultipliers,
        flow_multipliers: NetLearnRateMultipliers,
        reset_multipliers: NetLearnRateMultipliers,
    ):
        mapping = {
            SeriesType.FULL: full_multipliers,
            SeriesType.FLOW: flow_multipliers,
            SeriesType.RESET: reset_multipliers,
        }
        super().__init__(mapping)


class LearnRateObj:
    def __init__(
        self,
        itr_n: int = 0,
        epoch_n: int = 0,
        warmup_steps: int = 0,
        lr_base: float = 1.0,
        lr_mults: (None | SeriesLearnRateMultipliers) = None,
        exp_decay_coef: (float | None) = None,
        cos_decay_coef: (float | None) = None,
    ):
        if (exp_decay_coef is not None) and (cos_decay_coef is not None):
            raise ValueError("Cannot combine exponential and cosine decays")

        self.itr_n = itr_n
        self.epoch_n = epoch_n
        self.warmup_steps = warmup_steps
        self.lr_base = lr_base
        self.lr_mults = lr_mults
        # alpha is the floor above zero of the cosine decay function
        self.exp_decay = exp_decay_coef
        self.cos_alpha = cos_decay_coef
        self.decay_steps = (self.itr_n * self.epoch_n) - self.warmup_steps

    def get_lr_dict(self, current_step: int, traj_type: SeriesType):
        base_lr = self.lr_base
        if current_step > self.warmup_steps:
            post_warmup_step = current_step - self.warmup_steps
            if self.exp_decay is not None:
                base_lr = base_lr * (self.exp_decay**post_warmup_step)
            elif self.cos_alpha is not None and self.decay_steps > 0:
                shifted_cosine = 0.5 * (
                    1 + math.cos(math.pi * post_warmup_step / self.decay_steps)
                )
                alpha_decay = (1 - self.cos_alpha) * shifted_cosine + self.cos_alpha
                base_lr = base_lr * alpha_decay
        else:
            base_lr = (current_step / self.warmup_steps) * self.lr_base
        if self.lr_mults is None:
            return base_lr
        else:
            return {k: v * base_lr for k, v in self.lr_mults[traj_type].items()}

    def get_epoch_range(self, itr):
        return range(self.epoch_n * itr, self.epoch_n * (itr + 1))


class TrajLossObj:
    def __init__(
        self,
        n_state_variables: int,
        loss_power: float = 2,
        normalized_time_scaling_fn: (Callable[[torch.Tensor],torch.Tensor] | None) = None,
        traj_pos_weight=1.0,
        traj_vel_weight=1.0,
        traj_accel_weight=1.0,
    ):
        self.n_state_variables = n_state_variables
        self.loss_power = loss_power
        self.normalized_time_scaling_fn = normalized_time_scaling_fn
        self.traj_pos_weight = traj_pos_weight
        self.traj_vel_weight = traj_vel_weight
        self.traj_accel_weight = traj_accel_weight

    def scaled_loss(
        self,
        index_time: torch.Tensor,
        net_output: torch.Tensor,
        ground_truth: torch.Tensor,
    ):
        time_progress = index_time - index_time[0]
        time_progress = time_progress / time_progress[-1]
        loss = net_output - ground_truth
        loss = torch.pow(loss, self.loss_power)
        if self.loss_power % 2 != 0:
            loss = torch.abs(loss)
        if self.normalized_time_scaling_fn is not None:
            scale_factor = self.normalized_time_scaling_fn(time_progress)
            loss = torch.mul(loss, scale_factor)
        loss = torch.mean(loss, 1)
        return loss

    def calculate_loss(
        self,
        model: NNDynamicsObject,
        ic: torch.Tensor,
        data: torch.Tensor,
        local_state: bool,
    ):
        time = data[0]
        coord_data = data[1:]
        gt_traj = coord_data[: self.n_state_variables]
        gt_vel = coord_data[self.n_state_variables : self.n_state_variables * 2]
        if coord_data.size(0) > self.n_state_variables * 2:
            gt_accel = coord_data[self.n_state_variables * 2 :]
            _, model_traj, calc_accel, _ = model.simulate(
                time, return_accel=True, initial_state=ic, local_state=local_state
            )
        else:
            gt_accel = None
            _, model_traj, _, _ = model.simulate(
                time, initial_state=ic, local_state=local_state
            )

        calc_traj = model_traj[: self.n_state_variables]
        calc_vel = model_traj[self.n_state_variables :]

        pos_loss = torch.sum(
            torch.mul(self.scaled_loss(time, calc_traj, gt_traj), self.traj_pos_weight)
        )
        vel_loss = torch.sum(
            torch.mul(self.scaled_loss(time, calc_vel, gt_vel), self.traj_vel_weight)
        )

        if gt_accel is not None:
            accel_loss = torch.sum(
                torch.mul(
                    self.scaled_loss(time, calc_accel, gt_accel), self.traj_accel_weight
                )
            )
        else:
            accel_loss = torch.zeros_like(pos_loss)
        loss = pos_loss + vel_loss + accel_loss
        return loss


def get_param_groups(model: NNDynamicsObject):
    param_groups = [
        {
            "params": model.force_module.force_net.parameters()
            if hasattr(model.force_module, "force_net")
            else []
        },
        {
            "params": model.event_module.event_params
            if hasattr(model.event_module, "event_params")
            else []
        },
        {
            "params": model.reset_module.reset_net.parameters()
            if hasattr(model.reset_module, "reset_net")
            else []
        },
        {
            "params": model.dynamics_module.dynamics_net.parameters()
            if hasattr(model.dynamics_module, "dynamics_net")
            else []
        },
    ]
    return param_groups


def single_train(
    itr: int,
    device: torch.Device,
    model: NNDynamicsObject,
    optimizer: type[torch.optim.Optimizer],
    data_loader: (SingleThreadDataloader | MultiThreadDataloader),
    lr_obj: LearnRateObj,
    traj_loss_obj: TrajLossObj,
):
    for step_i in lr_obj.get_epoch_range(itr):
        print("epoch " + str(step_i) + " start")
        epoch_train(
            device, model, optimizer, data_loader, lr_obj, traj_loss_obj, step_i
        )
        print("epoch " + str(step_i) + " done")


def parallel_train(
    rank: int,
    itr: int,
    device: torch.Device,
    model: NNDynamicsObject,
    data_loader: MultiThreadDataloader,
    lr_obj: LearnRateObj,
    traj_loss_obj: TrajLossObj,
    lock: Lock,
):
    # model parameters - order is event parameters, reset parameters, and dynamics parameters
    param_groups = get_param_groups(model)
    parallel_optimizer = torch.optim.Adam(param_groups)

    for step_i in lr_obj.get_epoch_range(itr):
        print("thread " + str(rank) + " epoch " + str(step_i) + " start")
        epoch_train(
            device,
            model,
            parallel_optimizer,
            data_loader,
            lr_obj,
            traj_loss_obj,
            step_i,
            mp_lock=lock,
        )
        print("thread " + str(rank) + " epoch " + str(step_i) + " done")


def epoch_train(
    device: torch.Device,
    model: NNDynamicsObject,
    optimizer: type[torch.optim.Optimizer],
    data_loader: (SingleThreadDataloader | MultiThreadDataloader),
    lr_obj: LearnRateObj,
    traj_loss_obj: TrajLossObj,
    step_i: int,
    mp_lock: (Lock | None) = None,
):
    for data, label, ic in data_loader:
        lr_dict = lr_obj.get_lr_dict(step_i, label)
        if data.device != device:
            data = data.to(device, non_blocking=True)
            ic = ic.to(device, non_blocking=True)
        if label == SeriesType.FULL:
            optimizer.param_groups[0]["lr"] = lr_dict[NetType.FORCE]
            optimizer.param_groups[1]["lr"] = lr_dict[NetType.EVENT]
            optimizer.param_groups[2]["lr"] = lr_dict[NetType.RESET]
            optimizer.param_groups[3]["lr"] = lr_dict[NetType.DYNAMICS]
        elif label == SeriesType.FLOW:
            optimizer.param_groups[0]["lr"] = lr_dict[NetType.FORCE]
            optimizer.param_groups[1]["lr"] = 0
            optimizer.param_groups[2]["lr"] = 0
            optimizer.param_groups[3]["lr"] = lr_dict[NetType.DYNAMICS]
        elif label == SeriesType.RESET:
            optimizer.param_groups[0]["lr"] = 0
            optimizer.param_groups[1]["lr"] = lr_dict[NetType.EVENT]
            optimizer.param_groups[2]["lr"] = lr_dict[NetType.RESET]
            optimizer.param_groups[3]["lr"] = 0

        optimizer.zero_grad()

        loss = traj_loss_obj.calculate_loss(model, ic, data, (mp_lock is not None))

        if loss.requires_grad:
            loss.backward()
            if any(
                torch.any(torch.isnan(p.grad))
                for p in model.parameters()
                if p is not None and p.grad is not None
            ):
                print("nan parameter, loss = " + str(loss) + ", ic = " + str(ic))
                print("trajectory")
                print(
                    model.simulate(
                        data[0], initial_state=ic, local_state=mp_lock is not None
                    )
                )

            if mp_lock is not None:
                mp_lock.acquire()
                try:
                    optimizer.step()
                finally:
                    mp_lock.release()
            else:
                optimizer.step()
