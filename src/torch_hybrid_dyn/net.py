import math

import torch
import torch.nn.functional as TF
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim=0,
        hidden_depth=0,
        act=nn.LeakyReLU,
    ):
        super().__init__()
        if input_dim == 0:
            self.constant = True
            self.param = nn.Parameter(torch.rand(output_dim) - 0.5)
        else:
            self.constant = False
            if hidden_depth == 0:
                mods = [nn.Linear(input_dim, output_dim)]
            else:
                mods = [nn.Linear(input_dim, hidden_dim), act()]
                for _ in range(hidden_depth - 1):
                    mods.append(nn.Linear(hidden_dim, hidden_dim))
                    if act is not None:
                        mods.append(act())
                mods.append(nn.Linear(hidden_dim, output_dim))
            self.net = nn.Sequential(*mods)

    def zero_last_layer(self):
        if not self.constant:
            with torch.no_grad():
                *_, last_module = self.net
                last_module.weight.data = torch.zeros_like(last_module.weight.data)

    def forward(self, tensor_input) -> torch.Tensor:
        if self.constant:
            return self.param
        else:
            return self.net(tensor_input)


class SLLMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dims: list[int],
        hidden_dim=0,
        hidden_depth=0,
        divergence_depth=-1,
        act=nn.LeakyReLU,
    ):
        super().__init__()

        # no point having multiple layers of net if there's no input, so just learn each output as a constant in that case
        if input_dim == 0:
            self.constant = True
            self.params = nn.ParameterList(
                [
                    torch.rand(output_dim) - 0.5 if output_dim > 0 else None
                    for output_dim in output_dims
                ]
            )
        else:
            self.constant = False
            # if there are no hidden layers, just wire each output up to the appropriate number of inputs
            if hidden_dim == 0:
                self.nets = nn.ModuleList(
                    [
                        nn.Linear(input_dim, output_dim) if output_dim > 0 else None
                        for output_dim in output_dims
                    ]
                )
            else:
                # divergence depth can be between 0 and the number of hidden layers
                divergence_depth = max(divergence_depth, hidden_depth)
                if divergence_depth < 0:
                    divergence_depth = max(hidden_depth - divergence_depth, 0)
                # if divergence is supposed to be immediate and there are hidden layers, skip the preliminary stuff and
                # initialize the divergence list with an input map to the hidden layer size
                if divergence_depth == 0:
                    base = []
                    for _ in output_dims:
                        if act is None:
                            extensions = [
                                [nn.Linear(input_dim, hidden_dim)] for _ in output_dims
                            ]
                        else:
                            extensions = [
                                [nn.Linear(input_dim, hidden_dim), act()]
                                for _ in output_dims
                            ]
                else:
                    base = [nn.Linear(input_dim, hidden_dim)]
                    if act is not None:
                        base.append(act())
                    # if we have exactly one layer of hidden network that diverges on output wiring, special-case that
                    if hidden_depth == 1:
                        extensions = [
                            [nn.Linear(hidden_dim, output_dim)]
                            for output_dim in output_dims
                        ]
                    else:
                        # make converged layers up to the divergence depth
                        for _ in range(divergence_depth - 1):
                            base.append(nn.Linear(hidden_dim, hidden_dim))
                            if act is not None:
                                base.append(act())
                        # calculate remaining hidden layers and populate them for each output
                        remaining_hidden_layers = hidden_depth - divergence_depth
                        extensions = []
                        for output_dim in output_dims:
                            if output_dim > 0:
                                i_ext = []
                                for _ in range(remaining_hidden_layers):
                                    i_ext.append(nn.Linear(hidden_dim, hidden_dim))
                                    if act is not None:
                                        i_ext.append(act())
                                i_ext.append(nn.Linear(hidden_dim, output_dim))
                            else:
                                i_ext = None
                            extensions.append(i_ext)
                self.nets = nn.ModuleList(
                    [
                        nn.Sequential(*(base + i_ext)) if i_ext is not None else None
                        for i_ext in extensions
                    ]
                )

    def zero_last_layer(self):
        if not self.constant:
            with torch.no_grad():
                for net in self.nets:
                    if net is not None:
                        *_, last_module = net
                        last_module.weight.data = torch.zeros_like(
                            last_module.weight.data
                        )

    def forward(self, tensor_input: torch.Tensor, index: int) -> torch.Tensor:
        if self.constant:
            return self.params[index]
        else:
            if self.nets[index] is None:
                return None
            else:
                return self.nets[index](tensor_input)


class FunctionalMLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim=0, hidden_depth=0, act=nn.ReLU
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(
        self, tensor_input: torch.Tensor, model_params: torch.Tensor
    ) -> torch.Tensor:
        if self.hidden_depth == 0:
            net_result = TF.linear(
                tensor_input,
                torch.reshape(
                    model_params[: self.output_dim * self.input_dim],
                    (self.output_dim, self.input_dim),
                ),
                model_params[self.output_dim * self.input_dim :],
            )
        else:
            net_result = TF.linear(
                tensor_input,
                torch.reshape(
                    model_params[: self.hidden_dim * self.input_dim],
                    (self.hidden_dim, self.input_dim),
                ),
                model_params[
                    self.hidden_dim * self.input_dim : self.hidden_dim
                    * (self.input_dim + 1)
                ],
            )
            offset = self.hidden_dim * (self.input_dim + 1)
            for layer_idx in range(self.hidden_depth - 1):
                net_result = TF.linear(
                    net_result,
                    torch.reshape(
                        model_params[
                            offset : offset + self.hidden_dim * self.hidden_dim
                        ],
                        (self.hidden_dim, self.hidden_dim),
                    ),
                    model_params[
                        offset + self.hidden_dim * self.hidden_dim : offset
                        + self.hidden_dim * (self.hidden_dim + 1)
                    ],
                )
                net_result = self.act.forward(net_result)
                offset = offset + self.hidden_dim * (self.hidden_dim + 1)
            net_result = TF.linear(
                net_result,
                torch.reshape(
                    model_params[offset : offset + self.output_dim * self.hidden_dim],
                    (self.output_dim, self.hidden_dim),
                ),
                model_params[offset + self.output_dim * self.hidden_dim :],
            )
        return net_result

    def export_init_params(self) -> torch.Tensor:
        param_list = []
        input_size_sqrt = 2 * math.sqrt(self.input_dim)
        hidden_size_sqrt = 2 * math.sqrt(self.hidden_dim)
        if self.hidden_depth == 0:
            weight = torch.rand(self.output_dim * self.input_dim)
            bias = torch.rand(self.output_dim) * input_size_sqrt - input_size_sqrt / 2
            param_list.extend([weight, bias])
        else:
            weight = (
                torch.rand(self.hidden_dim * self.input_dim) * input_size_sqrt
                - input_size_sqrt / 2
            )
            bias = torch.rand(self.hidden_dim) * input_size_sqrt - input_size_sqrt / 2
            param_list.extend([weight, bias])
            for layer_idx in range(self.hidden_depth - 1):
                weight = (
                    torch.rand(self.hidden_dim * self.hidden_dim) * hidden_size_sqrt
                    - hidden_size_sqrt / 2
                )
                bias = (
                    torch.rand(self.hidden_dim) * hidden_size_sqrt
                    - hidden_size_sqrt / 2
                )
                param_list.extend([weight, bias])
            weight = torch.rand(self.output_dim * self.hidden_dim)
            bias = torch.rand(self.output_dim) * hidden_size_sqrt - hidden_size_sqrt / 2
            param_list.extend([weight, bias])
        return torch.cat(param_list)
