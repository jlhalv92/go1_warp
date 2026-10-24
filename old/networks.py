"""User-specified SiLU/LayerNorm actor and critic, with Gaussian exploration."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def _input(x):
    # Accept (B,D) and (B,1,D) without dropping B when B=1.
    return (x.squeeze(1) if x.ndim == 3 else x).float()


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input, n_output = input_shape[-1], output_shape[0]
        self.fc1 = nn.Linear(n_input, 2 * n_features)
        self.ln1 = nn.LayerNorm(2 * n_features)
        self.fc2 = nn.Linear(2 * n_features, n_features)
        self.ln2 = nn.LayerNorm(n_features)
        self.fc3 = nn.Linear(n_features, n_features // 2)
        self.ln3 = nn.LayerNorm(n_features // 2)
        self.fc4 = nn.Linear(n_features // 2, n_output)

    def forward(self, x):
        x = F.silu(self.ln1(self.fc1(_input(x))))
        x = F.silu(self.ln2(self.fc2(x)))
        x = F.silu(self.ln3(self.fc3(x)))
        return self.fc4(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        n_input, n_output = input_shape[-1], output_shape[0]
        self._h1 = nn.Linear(n_input, 2 * n_features)
        self.ln1 = nn.LayerNorm(2 * n_features)
        self._h2 = nn.Linear(2 * n_features, n_features)
        self.ln2 = nn.LayerNorm(n_features)
        self._h3 = nn.Linear(n_features, n_features // 2)
        self.ln3 = nn.LayerNorm(n_features // 2)
        self._h4 = nn.Linear(n_features // 2, n_output)
        for layer in (self._h1, self._h2, self._h3):
            nn.init.orthogonal_(layer.weight, np.sqrt(2))
            nn.init.constant_(layer.bias, 0.)
        nn.init.orthogonal_(self._h4.weight, .01)
        nn.init.constant_(self._h4.bias, 0.)

    def forward(self, state, **kwargs):
        x = F.silu(self.ln1(self._h1(_input(state))))
        x = F.silu(self.ln2(self._h2(x)))
        x = F.silu(self.ln3(self._h3(x)))
        return self._h4(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=48, action_dim=12, n_features=256, std_0=1.):
        super().__init__()
        if n_features < 2 or std_0 <= 0:
            raise ValueError('n_features >= 2 and std_0 > 0 are required')
        self.actor = PolicyNetwork((obs_dim,), (action_dim,), n_features)
        self.critic = ValueNetwork((obs_dim,), (1,), n_features)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(std_0)))

    def distribution(self, obs):
        mean = self.actor(obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def action_value(self, obs, action=None):
        dist = self.distribution(obs)
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1),
                self.value(obs), dist.loc, dist.scale)

    def entropy(self):
        return (self.log_std + .5 * math.log(2 * math.pi * math.e)).sum()
