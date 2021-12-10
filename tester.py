import collections

import gym
import numpy as np
import ray

from model import EmbeddingNet, LifeLongNet, QNetwork
from utils import UCB, create_beta_list, get_preprocess_func, play_episode


@ray.remote(num_cpus=1)
class Tester:

    def __init__(self,
                 env_name,
                 n_frames, num_arms,
                 L,
                 k,
                 window_size,
                 ucb_epsilon,
                 ucb_beta,
                 original_lifelong_weight):
        
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.frame_process_func = get_preprocess_func(env_name)
        self.n_frames = n_frames
        self.action_space = self.env.action_space.n

        self.in_q_network = QNetwork(self.action_space)
        self.ex_q_network = QNetwork(self.action_space)
        self.embedding_net = EmbeddingNet()
        self.original_lifelong_net = LifeLongNet()
        self.trained_lifelong_net = LifeLongNet()

        self.ucb = UCB(num_arms, window_size, ucb_epsilon, ucb_beta)
        self.betas = create_beta_list(num_arms)

        self.error_list = collections.deque(maxlen=int(1e5))
        self.L = L
        self.k = k

        self.original_lifelong_net.load_state_dict(original_lifelong_weight)
        self.is_test = True
        self.count = 0
        
    
    def test_play(self, in_q_weight, ex_q_weight, embed_weight, lifelong_weight):
        
        if self.count % 5 == 0:
            self.in_q_network.load_state_dict(in_q_weight)
            self.ex_q_network.load_state_dict(ex_q_weight)
            self.embedding_net.load_state_dict(embed_weight)
            self.trained_lifelong_net.load_state_dict(lifelong_weight)

        j = self.ucb.pull_index()
        beta = self.betas[j]
        
        if self.is_test:            
            _, episode_reward, self.error_list = play_episode(frame_process_func=self.frame_process_func,
                                                              env_name=self.env_name,
                                                              n_frames=self.n_frames,
                                                              action_space=self.action_space,
                                                              j=j,
                                                              epsilon=0.0,
                                                              k=self.k,
                                                              error_list=self.error_list,
                                                              L=self.L,
                                                              beta=beta,
                                                              in_q_network=self.in_q_network,
                                                              ex_q_network=self.ex_q_network,
                                                              embedding_net=self.embedding_net,
                                                              original_lifelong_net=self.original_lifelong_net,
                                                              trained_lifelong_net=self.trained_lifelong_net,
                                                              is_test=True)
            
            self.episode_reward.append(episode_reward)
            self.count += 1
        
        else:
            ucb_datas, _, self.error_list = play_episode(frame_process_func=self.frame_process_func,
                                                         env_name=self.env_name,
                                                         n_frames=self.n_frames,
                                                         action_space=self.action_space,
                                                         j=j,
                                                         epsilon=0.01,
                                                         k=self.k,
                                                         error_list=self.error_list,
                                                         L=self.L,
                                                         beta=beta,
                                                         in_q_network=self.in_q_network,
                                                         ex_q_network=self.ex_q_network,
                                                         embedding_net=self.embedding_net,
                                                         original_lifelong_net=self.original_lifelong_net,
                                                         trained_lifelong_net=self.trained_lifelong_net,
                                                         is_test=True)
            self.ucb.push_data(ucb_datas)
            self.count += 1       
            
        if self.count % 10 == 5:
            self.is_test = True
            self.episode_reward = []
            
        elif self.count % 10 == 0:
            self.is_test = False
            return np.mean(self.episode_reward)
