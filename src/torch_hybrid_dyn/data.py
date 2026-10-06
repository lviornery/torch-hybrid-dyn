import itertools
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .util import SeriesType


class DiffEQDataset(Dataset):
    def __init__(
        self,
        full_slice_data: list[torch.Tensor],
        flow_slice_data: list[torch.Tensor],
        reset_slice_data: list[torch.Tensor],
        full_slice_ics: list[torch.Tensor],
        flow_slice_ics: list[torch.Tensor],
        reset_slice_ics: list[torch.Tensor],
    ):
        self.slice_data = list(
            itertools.chain(full_slice_data, flow_slice_data, reset_slice_data)
        )
        self.slice_labels = list(
            itertools.chain(
                [SeriesType.FULL] * len(full_slice_data),
                [SeriesType.FLOW] * len(flow_slice_data),
                [SeriesType.RESET] * len(reset_slice_data),
            )
        )
        self.slice_ics = list(
            itertools.chain(full_slice_ics, flow_slice_ics, reset_slice_ics)
        )

    def __len__(self):
        return len(self.slice_labels)

    def __getitem__(self, idx):
        data = self.slice_data[idx]
        label = self.slice_labels[idx]
        ic = self.slice_ics[idx]
        return data, label, ic

    def add_item(self, new_data_list, new_label_list, new_ic_list):
        self.slice_data.extend(new_data_list)
        self.slice_labels.extend(new_label_list)
        self.slice_ics.extend(new_ic_list)

    def get_full_trajectories(self):
        return [
            i_data
            for i_data, i_label in zip(self.slice_data, self.slice_labels)
            if i_label == SeriesType.FULL
        ]

    def get_whole_ics(self):
        return [
            i_ic
            for i_ic, i_label in zip(self.slice_ics, self.slice_labels)
            if i_label == SeriesType.FULL
        ]


class SingleThreadDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset: DiffEQDataset):
        super().__init__(dataset, shuffle=True, batch_size=None)


class MultiThreadDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset: DiffEQDataset, threads: int, rank: int, seed: int):
        sampler = DiffeqDistributedSampler(dataset, threads, rank, seed)
        super().__init__(dataset, sampler=sampler, batch_size=None)


class DiffeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset: DiffEQDataset, threads: int, rank: int, seed: int):
        super().__init__(
            dataset,
            num_replicas=threads,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )


def generate_sliced_dataset(data: list[torch.Tensor], do_slice=False, slice_step=5):
    """returns a DiffEQDataset holding all of the input data.
    The first list is purely dynamic, the second is purely near resets, and the third is full trajectories
    Inputs: data, a list formatted as:
    [
        <n-length time tensor
        <nxm state tensor where m is the state dimension>
        <list of tensor times at which events occurred>
        <list of tensor hybrid states which were transitioned into at the event times>
        <an initial condition>
    ]"""
    full_slices: list[torch.Tensor] = []
    flow_slices: list[torch.Tensor] = []
    reset_slices: list[torch.Tensor] = []
    full_slice_ics: list[torch.Tensor] = []
    flow_slice_ics: list[torch.Tensor] = []
    reset_slice_ics: list[torch.Tensor] = []

    for i in range(len(data)):
        i_obs_times = data[i][0]
        i_gt_trajectory = data[i][1]
        i_hybrid_states = data[i][2]
        i_event_times = data[i][3]

        # full trajectory as a mixed trajectory no matter what
        full_slices.append(
            torch.cat((torch.unsqueeze(i_obs_times, 0), i_gt_trajectory))
        )
        full_slice_ics.append(torch.cat((i_gt_trajectory[:, 0], i_hybrid_states[0])))
        if do_slice:
            start_index = 0
            event_indices = torch.cat(
                (
                    np.searchsorted(
                        i_obs_times.cpu(), [et.cpu() for et in i_event_times]
                    ),
                    torch.tensor([len(i_obs_times)], device="cpu"),
                )
            )
            for hybrid_state_val,event_index in zip(i_hybrid_states,event_indices):
                event_index_post = None
                dyn_time_slice = None
                reset_time_slice = None

                # normal segment pair - dynamics flight followed by reset
                if event_index - slice_step > start_index + math.ceil(
                    slice_step / 2
                ) and event_index + slice_step < len(i_obs_times) - math.ceil(
                    slice_step / 2
                ):
                    event_index_pre = event_index - slice_step
                    event_index_post = event_index + slice_step

                    dyn_time_slice = i_obs_times[start_index:event_index_pre]
                    dyn_traj_slice = i_gt_trajectory[:, start_index:event_index_pre]

                    reset_time_slice = i_obs_times[event_index_pre:event_index_post]
                    reset_traj_slice = i_gt_trajectory[
                        :, event_index_pre:event_index_post
                    ]

                # no space for dynamic flight, just a reset
                elif event_index - slice_step <= start_index + math.ceil(
                    slice_step / 2
                ) and event_index + slice_step < len(i_obs_times):
                    event_index_post = event_index + slice_step

                    reset_time_slice = i_obs_times[start_index:event_index_post]
                    reset_traj_slice = i_gt_trajectory[:, start_index:event_index_post]

                # no space for post-reset, just dynamic flight
                elif event_index - 1 > start_index and event_index + slice_step >= len(
                    i_obs_times
                ) - math.ceil(slice_step / 2):
                    dyn_time_slice = i_obs_times[start_index : event_index - 1]
                    dyn_traj_slice = i_gt_trajectory[:, start_index : event_index - 1]

                if (
                    torch.is_tensor(dyn_time_slice)
                    and len(dyn_time_slice) >= slice_step / 2
                ):
                    flow_slices.append(
                        torch.cat((torch.unsqueeze(dyn_time_slice, 0), dyn_traj_slice))
                    )
                    flow_slice_ics.append(
                        torch.cat(
                            (dyn_traj_slice[:, 0], hybrid_state_val)
                        )
                    )
                if (
                    torch.is_tensor(reset_time_slice)
                    and len(reset_time_slice) >= slice_step / 2
                ):
                    reset_slices.append(
                        torch.cat(
                            (torch.unsqueeze(reset_time_slice, 0), reset_traj_slice)
                        )
                    )
                    reset_slice_ics.append(
                        torch.cat(
                            (reset_traj_slice[:, 0], hybrid_state_val)
                        )
                    )

                if event_index_post:
                    start_index = event_index_post

    dataset = DiffEQDataset(
        full_slices,
        flow_slices,
        reset_slices,
        full_slice_ics,
        flow_slice_ics,
        reset_slice_ics,
    )

    return dataset
