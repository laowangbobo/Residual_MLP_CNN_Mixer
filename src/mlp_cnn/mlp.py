# by Yang Haibo(2022/1/11)
# yanghaibo4869@gmail.com

import torch.nn as nn


'''
This code is used to build the residual mlp part of network.

The original code can be found in
https://github.com/Wenxuan-1119/TransBTS/blob/main/models/TransBTS/Transformer.py
'''


# Build cascaded structure of network and save the intermediate outputs
class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs

# Build residual structure of network.
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# build layernorm structure of network
class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))

# build mlp structure of network.
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

# build residual mlp structure of network 
class mlp_mix(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        mlp_dim,
        dropout_rate=0.1
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            FeedForward(dim, mlp_dim, dropout_rate),
                        )
                    )
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)

    def forward(self, x):
        return self.net(x)
