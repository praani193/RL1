import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

from LHW.rl.policies.base import Net

class Actor(Net):
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self):
        raise NotImplementedError

    def adapt_input_layer(self, new_state_dim, preserve_weights=True):
        """
        Adapt the first input layer to accept `new_state_dim` inputs.
        - Supports ModuleList-based actor_layers with nn.Linear or nn.LSTMCell as first element,
          also supports Linear_Actor with attribute `l1`.
        - If preserve_weights is True, copies overlapping weights from the old layer.
        - Updates `state_dim` attribute if present and reshapes obs_mean/obs_std if present.
        """
        replaced = False

        # Helper to get device
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        # 1) If there's an actor_layers ModuleList, adapt the first element
        if hasattr(self, 'actor_layers') and isinstance(self.actor_layers, nn.ModuleList) and len(self.actor_layers) > 0:
            first = self.actor_layers[0]
            # Linear case
            if isinstance(first, nn.Linear):
                old_in = first.in_features
                out_features = first.out_features
                if old_in != new_state_dim:
                    new_first = nn.Linear(new_state_dim, out_features)
                    # preserve overlapping weights
                    with torch.no_grad():
                        if preserve_weights:
                            copy_len = min(old_in, new_state_dim)
                            # copy overlapping columns
                            new_first.weight[:, :copy_len] = first.weight[:, :copy_len]
                            if first.bias is not None and new_first.bias is not None:
                                new_first.bias.copy_(first.bias)
                        else:
                            nn.init.xavier_uniform_(new_first.weight)
                            if new_first.bias is not None:
                                new_first.bias.zero_()
                    self.actor_layers[0] = new_first
                    replaced = True

            # LSTMCell case
            elif isinstance(first, nn.LSTMCell):
                old_in = first.input_size
                hidden_sz = first.hidden_size
                if old_in != new_state_dim:
                    new_first = nn.LSTMCell(new_state_dim, hidden_sz)
                    with torch.no_grad():
                        if preserve_weights:
                            # weight_ih has shape (4*hidden, input_size)
                            copy_in = min(old_in, new_state_dim)
                            try:
                                new_first.weight_ih[:, :copy_in] = first.weight_ih[:, :copy_in]
                                # copy hidden-hidden weights and biases fully
                                new_first.weight_hh.copy_(first.weight_hh)
                                new_first.bias_ih.copy_(first.bias_ih)
                                new_first.bias_hh.copy_(first.bias_hh)
                            except Exception:
                                # Fallback: reinitialize if exact shapes don't match
                                pass
                        else:
                            nn.init.xavier_uniform_(new_first.weight_ih)
                            nn.init.xavier_uniform_(new_first.weight_hh)
                            new_first.bias_ih.zero_()
                            new_first.bias_hh.zero_()
                    self.actor_layers[0] = new_first
                    replaced = True

        # 2) If this is the simple Linear_Actor with attribute l1
        if not replaced and hasattr(self, 'l1') and isinstance(self.l1, nn.Linear):
            first = self.l1
            old_in = first.in_features
            out_features = first.out_features
            if old_in != new_state_dim:
                new_first = nn.Linear(new_state_dim, out_features)
                with torch.no_grad():
                    if preserve_weights:
                        copy_len = min(old_in, new_state_dim)
                        new_first.weight[:, :copy_len] = first.weight[:, :copy_len]
                        if first.bias is not None and new_first.bias is not None:
                            new_first.bias.copy_(first.bias)
                    else:
                        nn.init.xavier_uniform_(new_first.weight)
                        if new_first.bias is not None:
                            new_first.bias.zero_()
                self.l1 = new_first
                replaced = True

        # 3) Update state_dim attribute if present
        if hasattr(self, 'state_dim'):
            try:
                self.state_dim = new_state_dim
            except Exception:
                pass

        # 4) Fix obs_mean / obs_std shapes if present (they might be scalars initially)
        # Convert to torch tensors on correct device
        if hasattr(self, 'obs_mean'):
            om = self.obs_mean
            if isinstance(om, torch.Tensor):
                if om.numel() != new_state_dim:
                    self.obs_mean = torch.zeros(new_state_dim, device=device)
            else:
                # was scalar number
                self.obs_mean = torch.zeros(new_state_dim, device=device)

        if hasattr(self, 'obs_std'):
            osd = self.obs_std
            if isinstance(osd, torch.Tensor):
                if osd.numel() != new_state_dim:
                    self.obs_std = torch.ones(new_state_dim, device=device)
            else:
                self.obs_std = torch.ones(new_state_dim, device=device)

        if replaced:
            print(f"[Actor.adapt_input_layer] adapted first layer -> new_state_dim={new_state_dim}")
        else:
            print(f"[Actor.adapt_input_layer] no replacement needed (maybe model already matches {new_state_dim})")


class Actor(Actor):
    def __init__(self):
        super(Actor, self).__init__()

    def forward(self):
        raise NotImplementedError


class Linear_Actor(Actor):
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super(Linear_Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

        self.action_dim = action_dim

        for p in self.parameters():
            p.data = torch.zeros(p.shape)

    def forward(self, state):
        a = self.l1(state)
        a = self.l2(a)
        return a


class FF_Actor(Actor):
    def __init__(self, state_dim, action_dim, layers=(256, 256), nonlinearity=F.relu):
        super(FF_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers)-1):
            self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
        self.network_out = nn.Linear(layers[-1], action_dim)

        self.action_dim = action_dim
        self.nonlinearity = nonlinearity

        self.initialize_parameters()

    def forward(self, state, deterministic=True):
        x = state
        for idx, layer in enumerate(self.actor_layers):
            x = self.nonlinearity(layer(x))

        action = torch.tanh(self.network_out(x))
        return action


class LSTM_Actor(Actor):
    def __init__(self, state_dim, action_dim, layers=(128, 128), nonlinearity=torch.tanh):
        super(LSTM_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
        for i in range(len(layers)-1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
        self.network_out = nn.Linear(layers[i-1], action_dim)

        self.action_dim = action_dim
        self.init_hidden_state()
        self.nonlinearity = nonlinearity

    def get_hidden_state(self):
        return self.hidden, self.cells

    def set_hidden_state(self, data):
        if len(data) != 2:
            print("Got invalid hidden state data.")
            exit(1)

        self.hidden, self.cells = data

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

    def forward(self, x, deterministic=True):
        dims = len(x.size())

        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))
            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])

        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.nonlinearity(self.network_out(x))

            if dims == 1:
                x = x.view(-1)

        action = self.network_out(x)
        return action


class Gaussian_FF_Actor(Actor):  # more consistent with other actor naming conventions
    def __init__(self, state_dim, action_dim, layers=(256, 256), nonlinearity=torch.nn.functional.relu,
                 init_std=0.2, learn_std=False, bounded=False, normc_init=True):
        super(Gaussian_FF_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, layers[0])]
        for i in range(len(layers)-1):
            self.actor_layers += [nn.Linear(layers[i], layers[i+1])]
        self.means = nn.Linear(layers[-1], action_dim)

        self.learn_std = learn_std
        if self.learn_std:
            self.stds = nn.Parameter(init_std * torch.ones(action_dim))
        else:
            self.stds = init_std * torch.ones(action_dim)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.nonlinearity = nonlinearity

        # Initialized to no input normalization, can be modified later
        # Keep as tensors for safe arithmetic later
        self.obs_std = torch.tensor(1.0)
        self.obs_mean = torch.tensor(0.0)

        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.bounded = bounded

        self.init_parameters()

    def init_parameters(self):
        if self.normc_init:
            self.apply(normc_fn)
            self.means.weight.data.mul_(0.01)

    def _get_dist_params(self, state):
        # Ensure obs mean/std are tensors on the right device and shape
        if isinstance(self.obs_mean, (int, float)):
            self.obs_mean = torch.tensor(self.obs_mean, device=state.device)
        if isinstance(self.obs_std, (int, float)):
            self.obs_std = torch.tensor(self.obs_std, device=state.device)

        # If obs_mean/std are scalars, broadcasting is fine. If they are vectors, shape must match.
        # The Actor.adapt_input_layer method will convert obs_mean/std to correct size when needed.
        if isinstance(self.obs_mean, torch.Tensor) and self.obs_mean.numel() != state.shape[-1]:
            self.obs_mean = torch.zeros(state.shape[-1], device=state.device)
            self.obs_std = torch.ones(state.shape[-1], device=state.device)
        state = (state - self.obs_mean) / self.obs_std

        x = state
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))
        mean = self.means(x)

        if self.bounded:
            mean = torch.tanh(mean)

        sd = torch.zeros_like(mean)
        if hasattr(self, 'stds'):
            sd = self.stds
        return mean, sd

    def forward(self, state, deterministic=True):
        mu, sd = self._get_dist_params(state)

        if not deterministic:
            action = torch.distributions.Normal(mu, sd).sample()
        else:
            action = mu

        return action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)


class Gaussian_LSTM_Actor(Actor):
    def __init__(self, state_dim, action_dim, layers=(128, 128), nonlinearity=F.tanh, normc_init=False,
                 init_std=0.2, learn_std=False):
        super(Gaussian_LSTM_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.LSTMCell(state_dim, layers[0])]
        for i in range(len(layers)-1):
            self.actor_layers += [nn.LSTMCell(layers[i], layers[i+1])]
        self.network_out = nn.Linear(layers[i-1], action_dim)

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.init_hidden_state()
        self.nonlinearity = nonlinearity

        # Initialized to no input normalization, can be modified later
        self.obs_std = torch.tensor(1.0)
        self.obs_mean = torch.tensor(0.0)

        self.learn_std = learn_std
        if self.learn_std:
            self.stds = nn.Parameter(init_std * torch.ones(action_dim))
        else:
            self.stds = init_std * torch.ones(action_dim)

        if normc_init:
            self.initialize_parameters()

        self.act = self.forward

    def _get_dist_params(self, state):
        # Ensure obs mean/std are tensors on the right device and shape
        if isinstance(self.obs_mean, (int, float)):
            self.obs_mean = torch.tensor(self.obs_mean, device=state.device)
        if isinstance(self.obs_std, (int, float)):
            self.obs_std = torch.tensor(self.obs_std, device=state.device)

        state = (state - self.obs_mean) / self.obs_std

        dims = len(state.size())

        x = state
        if dims == 3:  # if we get a batch of trajectories
            self.init_hidden_state(batch_size=x.size(1))
            action = []
            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack([x_t for x_t in y])

        else:
            if dims == 1:  # if we get a single timestep (if not, assume we got a batch of single timesteps)
                x = x.view(1, -1)

            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]

            if dims == 1:
                x = x.view(-1)

        mu = self.network_out(x)
        sd = self.stds
        return mu, sd

    def init_hidden_state(self, batch_size=1):
        self.hidden = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size) for l in self.actor_layers]

    def forward(self, state, deterministic=True):
        mu, sd = self._get_dist_params(state)

        if not deterministic:
            action = torch.distributions.Normal(mu, sd).sample()
        else:
            action = mu

        return action

    def distribution(self, inputs):
        mu, sd = self._get_dist_params(inputs)
        return torch.distributions.Normal(mu, sd)


# Initialization scheme for gaussian mlp (from ppo paper)
# NOTE: the fact that this has the same name as a parameter caused a NASTY bug
# apparently "if <function_name>" evaluates to True in python...
def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
