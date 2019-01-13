
# Author: Christian Motschenbacher
# Date: 01/2019
# Code partially from teaching lessons

import torch
import torch.nn as nn
import torch.nn.functional as F

# DQN baseline network model
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,         64)
        self.fc3 = nn.Linear(64,         action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN baseline network model with he initialzation
class QNetwork_he_init(nn.Module):
    """Actor (Policy) Model with he initialization."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model
        initialised with he initialisation, because 
        of relu activation function.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork_he_init, self).__init__()
        # set random seed
        self.seed = torch.manual_seed(seed)
        # create three linear layers
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64,         64)
        self.fc3 = nn.Linear(64,         action_size)
        # initialize the layers with HE weights, because of the 
        # relu activation
        torch.nn.init.kaiming_uniform_(self.fc1.weight,nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight,nonlinearity='relu')

    def forward(self, state):
        """ Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN baseline network model with additional fully connected layers
class QNetwork_add_fcl(nn.Module):
    """Actor (Policy) Model with additional fully connected layer 
    in comparision to the base line model QNetwork."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork_add_fcl, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

# Dueling DQN network model
class QNetwork_dueling(nn.Module):
    """Dueling Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork_dueling, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        
        self.fc1_adv = nn.Linear(32, 16)
        self.fc2_adv = nn.Linear(16, action_size)
            
        self.fc1_val = nn.Linear(32, 16)
        self.fc2_val = nn.Linear(16, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        fc1_adv = F.relu(self.fc1_adv(x))
        fc1_val = F.relu(self.fc1_val(x))
        
        fc2_adv = self.fc2_adv(fc1_adv)
        fc2_val = self.fc2_val(fc1_val)
        
        return_val = fc2_val + fc2_adv - fc2_adv.mean()
        return return_val 

