import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolutional Encording
class ConvEncoder(nn.Module):
    """
    Encoder with convolution
    Attributes:
      conv1   : first convolutional layer
      conv2   : second convolutional layer 
      conv3   : third convolutional layer
      flatten : for tensor to be flatten
      fc      : full connected layer
    """
    def __init__(self, units, n_frames, hidden_list=[32, 64, 64], kernel_list=[8, 4, 3], stride_list=[4, 2, 1], flatten_dim=3136):
        super(ConvEncoder, self).__init__()
        """
        units    (int): dim of output
        n_frames (int): dim of input
        """
        
        self.conv1 = nn.Conv2d(n_frames, hidden_list[0], kernel_list[0], stride=stride_list[0])
        self.conv2 = nn.Conv2d(hidden_list[0], hidden_list[1], kernel_list[1], stride=stride_list[1])
        self.conv3 = nn.Conv2d(hidden_list[1], hidden_list[2], kernel_list[2], stride=stride_list[2])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flatten_dim, units)

    def forward(self, x):
        """
        Args:
          x (torch.tensor): input [b, n_frames, 84, 84]
        Returns
          x (torch.tensor): ouput [b, units]
        """
        
        x = F.relu(self.conv1(x))  # (b, 32, 20, 20)
        x = F.relu(self.conv2(x))  # (b, 64, 9, 9)
        x = F.relu(self.conv3(x))  # (b, 64, 7, 7)
        x = self.flatten(x)        # (b, 3136)
        x = self.fc(x)             # (b, units)
        return x

# intrinsic or extrinsic QNetwork
class QNetwork(nn.Module):
    """
    Attributes:
      action_space (int): dim of action space
      num_arms     (int): number of arms used in multi-armed bandit problem
      conv_encoder      : convolutional encoder
      lstm              : LSTM layer
      fc                : fully connected layer
      fc_adv            : fully connected layer to get advantage
      fc_v              : fully connected layer to get value
    """
    def __init__(self, action_space, n_frames, hidden=512, units=512, num_arms=32):
        super(QNetwork, self).__init__()
        """
        Args:
          action_space (int): dim of action space
          n_frames     (int): number of images to be stacked
        """
        
        self.action_space = action_space
        self.num_arms = num_arms

        self.conv_encoder = ConvEncoder(units, n_frames)
        self.lstm = nn.LSTM(input_size=units+self.action_space+self.num_arms+2,
                            hidden_size=hidden,
                            batch_first=False)
        
        self.fc = nn.Linear(hidden, hidden)
        self.fc_adv = nn.Linear(hidden, action_space)
        self.fc_v = nn.Linear(hidden, 1)

    def forward(self, input, states, prev_action, prev_in_rewards, prev_ex_rewards, j):
        """
        Args:
          input           (torch.tensor): state [b, n_frames, 84, 84]
          prev_action     (torch.tensor): previous action [b]
          prev_in_rewards (torch.tensor): previous intrinsic reward [b]
          prev_ex_rewards (torch.tensor): previous extrinsic reward [b]
        """
        
        # (b, q_units)
        x = F.relu(self.conv_encoder(input))

        # (b, action_space)
        prev_action_onehot = F.one_hot(prev_action, num_classes=self.action_space)
        
        # (b, num_arms)
        j_onehot = F.one_hot(j, num_classes=self.num_arms)
        
        # (b, q_units+action_space+num_arms+2)
        x = torch.cat([x, prev_action_onehot, prev_in_rewards[:, None], prev_ex_rewards[:, None], j_onehot], dim=1)

        # (1, b, hidden)
        x, states = self.lstm(x.unsqueeze(0), states)

        # (b, action_space)
        A = self.fc_adv(x.squeeze(0))
        
        # (b, 1)
        V = self.fc_v(x.squeeze(0))
        
        # (b, action_space)
        Q = V.expand(-1, self.action_space) + A - A.mean(1, keepdim=True).expand(-1, self.action_space)

        return Q, states


class EmbeddingNet(nn.Module):
    """
    Attributes
      conv_encoder : convolutional encoder
    """
    def __init__(self, n_frames, units=32):
        super(EmbeddingNet, self).__init__()
        """
        Args:
          n_frames (int): number of images to be stacked
        """
        
        self.conv_encoder = ConvEncoder(units, n_frames)

    def forward(self, inputs):
        """
        Args:
          input (torch.tensor): state [b, n_frames, 84, 84]
        Returns:
          embeded state [b, emebed_units]
        """
        
        return F.relu(self.conv_encoder(inputs))


class EmbeddingClassifer(nn.Module):
    """
    Attributes:
      fc1 : fully connected layer
      fc2 : fully connected layer to get action probability
    """
    def __init__(self, action_space, hidden=128):
        super(EmbeddingClassifer, self).__init__()
        """
        Args:
          action_space (int): dim of action space
        """
        
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, action_space)

    def forward(self, input1, input2):
        """
        Args:
          embeded state (torch.tensor): state [b, emebed_units]
        Returns:
          action probability [b, action_space]
        """
        
        x = torch.cat([input1, input2], dim=1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        
        return x


class LifeLongNet(nn.Module):
    """
    Attributes
      conv_encoder : convolutional encoder
    """
    def __init__(self, n_frames, units=128):
        super(LifeLongNet, self).__init__()
        """
        Args:
          n_frames (int): number of images to be stacked
        """
        
        self.conv_encoder = ConvEncoder(units, n_frames)

    def forward(self, inputs):
        """
        Args:
          input (torch.tensor): state [b, n_frames, 84, 84]
        Returns:
          lifelong state [b, lifelong_units]
        """
        
        return self.conv_encoder(inputs)
