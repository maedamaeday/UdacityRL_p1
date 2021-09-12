import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, n_hidden_units=[64, 16]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc_layers = nn.ModuleList()
        for i_layer in range(len(n_hidden_units)):
            self.fc_layers.append(
                nn.Linear(
                    state_size if i_layer==0 else n_hidden_units[i_layer-1],
                    n_hidden_units[i_layer],
                )
            )
        self.fc_layers.append( nn.Linear(n_hidden_units[-1],action_size) )
        self.relu = nn.ReLU()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        for i_layer, fc_layer in enumerate(self.fc_layers):
            if i_layer==0:
                out = fc_layer(state)
            else:
                out = fc_layer(out)
            if i_layer<len(self.fc_layers)-1:
                out = self.relu(out)
        
        return out
