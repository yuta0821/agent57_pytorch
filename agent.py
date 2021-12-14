import pickle
import collections

import gym
import lz4.frame as lz4f
import ray
import torch
import torch.nn.functional as F

from buffer import EpisodeBuffer
from model import EmbeddingNet, LifeLongNet, QNetwork
from utils import (UCB, create_beta_list, create_gamma_list,
                   get_preprocess_func, inverse_rescaling, play_episode,
                   rescaling, segments2contents, transformed_retrace_operator)


@ray.remote(num_cpus=1)
class Agent:
    def __init__(self,
                 pid,
                 env_name,
                 n_frames,
                 epsilon,
                 eta,
                 lamda,
                 num_arms,
                 burnin_length,
                 unroll_length,
                 k,
                 L,
                 agent_update_period,
                 num_rollout,
                 window_size,
                 ucb_epsilon,
                 ucb_beta,
                 original_lifelong_weight):
        
        self.pid = pid
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space.n
        self.frame_process_func = get_preprocess_func(self.env_name)
        self.n_frames = n_frames  

        self.in_q_network = QNetwork(self.action_space, n_frames)
        self.ex_q_network = QNetwork(self.action_space, n_frames)
        self.embedding_net = EmbeddingNet(n_frames)
        self.original_lifelong_net = LifeLongNet(n_frames)
        self.trained_lifelong_net = LifeLongNet(n_frames)

        self.ucb = UCB(num_arms, window_size, ucb_epsilon, ucb_beta)
        self.betas = create_beta_list(num_arms)
        self.gammas = create_gamma_list(num_arms)

        self.epsilon = epsilon
        self.eta = eta
        self.lamda = lamda

        self.burnin_len = burnin_length
        self.unroll_len = unroll_length
        
        self.k = k
        self.error_list = collections.deque(maxlen=int(1e4))
        self.L = L
        
        self.agent_update_period = agent_update_period
        self.num_rollout = num_rollout
        
        self.original_lifelong_net.load_state_dict(original_lifelong_weight)
        
        self.num_updates = 0

    def sync_weights_and_rollout(self, in_q_weight, ex_q_weight, embed_weight, lifelong_weight):

        if self.num_updates % self.agent_update_period == 0:
            self.in_q_network.load_state_dict(in_q_weight)
            self.ex_q_network.load_state_dict(ex_q_weight)
            self.embedding_net.load_state_dict(embed_weight)
            self.trained_lifelong_net.load_state_dict(lifelong_weight)

        # rollout 10steps
        priorities, segments = [], []
        while len(segments) < self.num_rollout:
            _priorities, _segments = self._rollout()
            priorities += _priorities
            segments += _segments

        self.num_updates += 1
        return priorities, segments, self.pid
    

    def _rollout(self):
        
        # get index from ucb
        j = self.ucb.pull_index()
        
        # get beta gamma
        beta, self.gamma = self.betas[j], self.gammas[j]

        episode_buffer = EpisodeBuffer(burnin_length=self.burnin_len, unroll_length=self.unroll_len)

        ucb_datas, transitions, self.error_list = play_episode(frame_process_func=self.frame_process_func,
                                                               env_name=self.env_name,
                                                               n_frames=self.n_frames,
                                                               action_space=self.action_space,
                                                               j=j,
                                                               epsilon=self.epsilon,
                                                               k=self.k,
                                                               error_list=self.error_list,
                                                               L=self.L,
                                                               beta=beta,
                                                               in_q_network=self.in_q_network,
                                                               ex_q_network=self.ex_q_network,
                                                               embedding_net=self.embedding_net,
                                                               original_lifelong_net=self.original_lifelong_net,
                                                               trained_lifelong_net=self.trained_lifelong_net)

        self.ucb.push_data(ucb_datas)

        for transition in transitions:
            episode_buffer.add(transition)

        # 初期優先度の計算 batch_sizeはsegmentsの長さによる
        segments = episode_buffer.pull_segments()
        
        self.states, self.actions, self.in_rewards, self.ex_rewards, self.dones, self.j, self.next_states, \
            in_h0, in_c0, ex_h0, ex_c0, self.prev_in_rewards, self.prev_ex_rewards, self.prev_actions = segments2contents(segments, self.burnin_len)

        # (unroll_len+1, batch_size, action_space)
        in_qvalues = self.get_qvalues(self.in_q_network, in_h0, in_c0)
        
        # (unroll_len+1, batch_size, action_space)
        ex_qvalues = self.get_qvalues(self.ex_q_network, ex_h0, ex_c0)

        # (unroll_len+1, batch_size)
        self.pi = torch.argmax(rescaling(inverse_rescaling(ex_qvalues) + beta * inverse_rescaling(in_qvalues)), dim=2)

        # (unroll_len, batch_size, action_space)
        self.actions_onehot = F.one_hot(self.actions[self.burnin_len:], num_classes=self.action_space)

        in_priorities = self.get_priorites(in_qvalues, self.in_rewards)
        ex_priorities = self.get_priorites(ex_qvalues, self.ex_rewards)

        priorities = in_priorities + ex_priorities
        compressed_segments = [lz4f.compress(pickle.dumps(seg)) for seg in segments]
        
        return priorities.detach().numpy().tolist(), compressed_segments

    def get_qvalues(self, q_network, h, c):
        
        for t in range(self.burnin_len):
            _, (h, c) = q_network(self.states[t],
                                   states=(h, c),
                                   prev_action=self.prev_actions[t],
                                   j=self.j,
                                   prev_in_rewards=self.prev_in_rewards[t],
                                   prev_ex_rewards=self.prev_ex_rewards[t])

        qvalues = []
        for t in range(self.burnin_len-1, self.burnin_len+self.unroll_len):
            
            # (batch_size, action_space)
            qvalue, (h, c) = q_network(self.next_states[t],
                                       states=(h, c),
                                       prev_action=self.actions[t],
                                       j=self.j,
                                       prev_in_rewards=self.in_rewards[t],
                                       prev_ex_rewards=self.ex_rewards[t])
            qvalues.append(qvalue)

        # (unroll_len+1, batch_size, action_space)
        qvalues = torch.stack(qvalues, dim=0)
        
        return qvalues

    def get_priorites(self, qvalues, rewards):
        
        # (unroll_len, batch_size)
        Q = torch.sum(qvalues[:-1] * self.actions_onehot, dim=2)
        
        # (unroll_len, batch_size)
        next_actions = torch.argmax(qvalues[1:], dim=2)

        # (unroll_len, batch_size, action_space)
        next_actions_onehot = F.one_hot(next_actions, self.action_space)

        # (unroll_len, batch_size)
        next_maxQ = torch.sum(qvalues[1:] * next_actions_onehot, dim=2)

        # (unroll_len, batch_size)
        TQ = rewards[self.burnin_len:] + self.gamma * (1 - self.dones) * inverse_rescaling(next_maxQ)

        # (unroll_len, batch_size)
        delta = TQ - inverse_rescaling(Q)

        # (unroll_len, batch_size)
        P = transformed_retrace_operator(delta,
                                         pi=self.pi[:-1],
                                         actions=self.actions[self.burnin_len:],
                                         lamda=self.lamda,
                                         gamma=self.gamma.repeat(delta.shape[1]),
                                         unroll_len=self.unroll_len)

        td_errors = rescaling(inverse_rescaling(Q) + P) - Q
        priorities = self.eta * torch.max(torch.abs(td_errors), dim=0).values + (1 - self.eta) * torch.mean(torch.abs(td_errors), dim=0)
        
        return  priorities


