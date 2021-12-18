import collections

import gym
import numpy as np
import ray

from model import EmbeddingNet, LifeLongNet, QNetwork
from utils import UCB, create_beta_list, get_preprocess_func, play_episode


@ray.remote(num_cpus=1)
class Tester:
    """
    calculate score to evaluate peformance
    Attributes:
      env_name          (str): name of environment
      n_frames          (int): number of images to be stacked
      env        (gym object): environment
      action_space      (int): dim of action space
      frame_process_func     : function to preprocess images
      in_q_network           : q network about intrinsic reward
      ex_q_network           : q network about extrinsic reward
      embedding_net          : embedding network to get episodic reward
      embedding_classifier   : classify action based on embedding representation
      original_lifelong_net  : lifelong network not to be trained
      trained_lifelong_net   : lifelong network to be trained
      ucb                    : object of UCB class which solve a multi-armed bandit problem
      betas            (list): list of beta which decide weights between intrinsic qvalues and extrinsic qvalues
      k                 (int): number of neighbors referenced when calculating episode reward
      L                 (int): upper limit of curiosity
      error_list             : list of errors to be accommodated when calculating lifelong reward
      switch_test_cycle (int): how often to switch test cycle from collecting ucb data cycle
      is_test          (bool): flag indicating whether it is a test or not
      count             (int): number of times to play test
    """

    def __init__(self,
                 env_name,
                 n_frames,
                 k,
                 L,
                 num_arms,
                 window_size,
                 ucb_epsilon,
                 ucb_beta,
                 switch_test_cycle,
                 original_lifelong_weight):
        """
        Args:
          env_name            (str): name of environment
          n_frames            (int): number of images to be stacked
          k                   (int): number of neighbors referenced when calculating episode reward
          L                   (int): upper limit of curiosity
          num_arms            (int): number of arms used in multi-armed bandit problem
          window_size         (int): size of window used in multi-armed bandit problem
          ucb_epsilon       (float): probability to select randomly used in multi-armed bandit problem
          ucb_beta          (float): weight between frequency and mean reward
          switch_test_cycle   (int): how often to switch test cycle from collecting ucb data cycle
          original_lifelong_weight : original weight of lifelong network 
        """
        
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.frame_process_func = get_preprocess_func(env_name)
        self.n_frames = n_frames
        self.action_space = self.env.action_space.n

        self.in_q_network = QNetwork(self.action_space, n_frames)
        self.ex_q_network = QNetwork(self.action_space, n_frames)
        self.embedding_net = EmbeddingNet(n_frames)
        self.original_lifelong_net = LifeLongNet(n_frames)
        self.trained_lifelong_net = LifeLongNet(n_frames)

        self.ucb = UCB(num_arms, window_size, ucb_epsilon, ucb_beta)
        self.betas = create_beta_list(num_arms)

        self.error_list = collections.deque(maxlen=int(1e4))
        self.k = k
        self.L = L
        
        self.switch_test_cycle = switch_test_cycle

        self.original_lifelong_net.load_state_dict(original_lifelong_weight)
        self.is_test = False
        self.count = 0
        
    
    def test_play(self, in_q_weight, ex_q_weight, embed_weight, lifelong_weight):
        """
        load weight and get score which is average of episode rewards
        Args:
          in_q_weight                              : weight of intrinsic q network
          ex_q_weight                              : weight of extrinsic q network
          embed_weight                             : weight of embedding network
          lifelong_weight                          : weight of lifelong network
        Returns:
          np.mean(self.episode_reward) (np.ndarray): average of episode rewards while test
        """
        
        if self.count % (self.switch_test_cycle//2) == 0:
            self.in_q_network.load_state_dict(in_q_weight)
            self.ex_q_network.load_state_dict(ex_q_weight)
            self.embedding_net.load_state_dict(embed_weight)
            self.trained_lifelong_net.load_state_dict(lifelong_weight)

        j = self.ucb.pull_index()
        beta = self.betas[j]
        
        # get episode reward
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
        
        # restore ucb datas
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
            
        if self.count % self.switch_test_cycle == (self.switch_test_cycle//2):
            self.is_test = True
            self.episode_reward = []
            
        elif self.count % self.switch_test_cycle == 0:
            self.is_test = False
            return np.mean(self.episode_reward)
