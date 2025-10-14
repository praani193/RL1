import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sqrt

def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.welford_state_mean = torch.zeros(1)
        self.welford_state_mean_diff = torch.ones(1)
        self.welford_state_n = 1

    def forward(self):
        raise NotImplementedError

    def normalize_state(self, state, update=True):
        state = torch.as_tensor(state, dtype=torch.float32)

        # Initialize stats for new state dimension
        if self.welford_state_n == 1:
            self.welford_state_mean = torch.zeros(state.size(-1))
            self.welford_state_mean_diff = torch.ones(state.size(-1))

        # Welford running mean/variance update
        if update:
            if state.ndim == 1:
                delta = state - self.welford_state_mean
                self.welford_state_mean += delta / self.welford_state_n
                self.welford_state_mean_diff += delta * (state - self.welford_state_mean)
                self.welford_state_n += 1
            elif state.ndim == 2:
                for state_n in state:
                    delta = state_n - self.welford_state_mean
                    self.welford_state_mean += delta / self.welford_state_n
                    self.welford_state_mean_diff += delta * (state_n - self.welford_state_mean)
                    self.welford_state_n += 1
            elif state.ndim == 3:
                for seq in state:
                    for state_n in seq:
                        delta = state_n - self.welford_state_mean
                        self.welford_state_mean += delta / self.welford_state_n
                        self.welford_state_mean_diff += delta * (state_n - self.welford_state_mean)
                        self.welford_state_n += 1

        std = sqrt(self.welford_state_mean_diff / self.welford_state_n)
        return (state - self.welford_state_mean) / (std + 1e-8)

    def copy_normalizer_stats(self, net):
        self.welford_state_mean = net.welford_state_mean.clone()
        self.welford_state_mean_diff = net.welford_state_mean_diff.clone()
        self.welford_state_n = net.welford_state_n

    def initialize_parameters(self):
        self.apply(normc_fn)
