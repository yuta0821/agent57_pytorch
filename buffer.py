import collections

import numpy as np

from segment_tree import SumTree

Transition = collections.namedtuple("Transition",
                                    ["prev_in_reward", "prev_ex_reward", "prev_action",
                                     "state", "action", "in_h", "in_c", "ex_h", "ex_c", "j",
                                     "done", "in_reward", "ex_reward", "next_state"])
Segment = collections.namedtuple("Segment",
                                 ["in_rewards", "ex_rewards", "states", "actions", "dones",
                                  "in_h_init", "in_c_init", "ex_h_init", "ex_c_init", "prev_a_init",
                                  "prev_in_reward_init", "prev_ex_reward_init", "last_state", "j"])


class EpisodeBuffer:
    """
    Attributes
      transitions [list]: list to store experiments
      burnin_len   [int]: length of burnin to calculate qvalues
      unroll_len   [int]: length of unroll to calculate qvalues
    """

    def __init__(self, burnin_length, unroll_length):
        """
        Args:
          burnin_len [int]: length of burnin to calculate qvalues
          unroll_len [int]: length of unroll to calculate qvalues
        """
        
        self.transitions = []
        self.burnin_len = burnin_length
        self.unroll_len = unroll_length

    def __len__(self):
        """
        set length
        """

        return len(self.transitions)

    def add(self, transition):
        """
        add transition (experience) to transitions
        Args:
          transition : experience collected by multi agents
        """
        
        transition = Transition(*transition)
        self.transitions.append(transition)

    def pull_segments(self):
        """
        generate segments from transitions
        Returns
          segments [list]: group of continues experiences
        """
        
        segments = []

        for t in range(self.burnin_len, len(self.transitions), self.unroll_len):
            if (t + self.unroll_len) > len(self.transitions):
                total_len = self.burnin_len + self.unroll_len
                timesteps = self.transitions[-total_len:]
            else:
                timesteps = self.transitions[t-self.burnin_len:t+self.unroll_len]

            segment = Segment(in_rewards=[t.in_reward for t in timesteps],
                              ex_rewards=[t.ex_reward for t in timesteps],
                              states=[t.state for t in timesteps],
                              actions=[t.action for t in timesteps],
                              dones=[t.done for t in timesteps],
                              in_h_init=timesteps[0].in_h,
                              in_c_init=timesteps[0].in_c,
                              ex_h_init=timesteps[0].ex_h,
                              ex_c_init=timesteps[0].ex_c,
                              prev_a_init=timesteps[0].prev_action,
                              prev_in_reward_init=timesteps[0].prev_in_reward,
                              prev_ex_reward_init=timesteps[0].prev_ex_reward,
                              last_state=timesteps[-1].next_state,
                              j=timesteps[0].j)
            segments.append(segment)
        return segments


class SegmentReplayBuffer:
    """
    Attributes
      buffer_size     [int]: size of buffer
      priorities           : SumTree object to determine priority
      segment_buffer [list]: buffer of segment which size is buffer_size
      weight_expo   [float]: exponetial value to smooth weights
      eta [float]          : coefficient for reduce priority
      count [int]          : index of priorities list
      full [bool]          : flag whether segment buffer is full or not
    """

    def __init__(self, buffer_size, weight_expo, eta=0.9):
        """
        Args
          buffer_size    [int]: size of buffer
          weight_expo  [float]: exponetial value to smooth weights
          eta [float, Optical]: coefficient for reduce priority
        """
        
        self.buffer_size = buffer_size
        self.priorities = SumTree(capacity=self.buffer_size)
        self.segment_buffer = [None] * self.buffer_size

        self.weight_expo = weight_expo
        self.eta = eta
        self.count = 0
        self.full = False

    def __len__(self):
        """
        set length
        """

        return len(self.segment_buffer) if self.full else self.count

    def add(self, priorities, segments):
        """
        add segment and priority to segment buffer
        Args:
          priorities [list]: list of priority, how much computed loss using the segment is 
          segments   [list]: group of continues experiences
        """
        
        assert len(priorities) == len(segments)

        for priority, segment in zip(priorities, segments):
            self.priorities[self.count] = priority
            self.segment_buffer[self.count] = segment

            self.count += 1
            if self.count == self.buffer_size:
                self.count = 0
                self.full = True

    def update_priority(self, sampled_indices, priorities):
        """
        update priority which is selected as a part of minibatch        
        Args:
          sampled_indices [list]: indices of segments which were selected as minibatch
          priorities      [list]: list of priority, how much computed loss using the segment is 
        """
        assert len(sampled_indices) == len(priorities)

        for idx, priority in zip(sampled_indices, priorities):
            self.priorities[idx] = priority ** self.eta

    def sample_minibatch(self, batch_size):
        """
        sample minibatch according to priorities
        Args:
          batch_size [int]: size of minibatch
        Returns:
          sampled_indices  [list]: indices of experiences
          sampled_weights  [list]: priorities of experiences
          sampled_segments [list]: a coherent body of experience of some length
        """
        
        sampled_indices = [self.priorities.sample() for _ in range(batch_size)]

        sampled_weights = []
        current_size = len(self.segment_buffer) if self.full else self.count
        
        for idx in sampled_indices:
            prob = self.priorities[idx] / self.priorities.sum()
            sampled_weight = (prob * current_size)**(-self.weight_expo)
            sampled_weights.append(sampled_weight)
            
        sampled_weights = np.array(sampled_weights) / max(sampled_weights)
        sampled_segments = [self.segment_buffer[idx] for idx in sampled_indices]
        
        return sampled_indices, sampled_weights, sampled_segments
