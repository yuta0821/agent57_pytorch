import os
import pickle
from concurrent import futures

import gym
import lz4.frame as lz4f
import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import EmbeddingClassifer, EmbeddingNet, LifeLongNet, QNetwork
from utils import (create_beta_list, create_gamma_list, get_preprocess_func,
                   inverse_rescaling, rescaling, segments2contents,
                   transformed_retrace_operator)


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    """
    update parameter
    Attributes:
      env_name             (str): name of environment
      n_frames             (int): number of images to be stacked
      env           (gym object): environment
      action_space         (int): dim of action space
      device      (torch.device): device to use
      frame_process_func        : function to preprocess images
      in_online_q_network       : online q network about intrinsic reward
      in_target_q_network       : target q network about intrinsic reward
      ex_online_q_network       : online q network about extrinsic reward
      ex_target_q_network       : target q network about extrinsic reward
      embedding_net             : embedding network to get episodic reward
      embedding_classifier      : classify action based on embedding representation
      original_lifelong_net     : lifelong network not to be trained
      trained_lifelong_net      : lifelong network to be trained
      in_q_optimizer            : optimizer of in_online_q_network
      ex_q_optimizer            : optimizer of ex_online_q_network
      embedding_optimizer       : optimizer of embedding_net
      ex_q_optimizer            : optimizer of trained_lifelong_net
      criterion                 : loss function of embedding classifier
      betas               (list): list of beta which decide weights between intrinsic qvalues and extrinsic qvalues
      gammas              (list): list of gamma which is discount rate
      eta                (float): coefficient for priority caluclation
      lamda              (float): coefficient for retrace operation
      burnin_length        (int): length of burnin to calculate qvalues
      unroll_length        (int): length of unroll to calculate qvalues
      target_update_period (int): how often to update the target parameters
      num_updates          (int): number of times to be updated
    """

    def __init__(self,
                 env_name,
                 n_frames,
                 eta,
                 lamda,
                 num_arms,
                 burnin_length,
                 unroll_length,
                 target_update_period,
                 in_q_lr,
                 ex_q_lr,
                 embed_lr,
                 lifelong_lr,
                 in_q_clip_grad,
                 ex_q_clip_grad,
                 embed_clip_grad,
                 lifelong_clip_grad):
        """
        Args:
          env_name             (str): name of environment
          n_frames             (int): number of images to be stacked
          eta                (float): coefficient for priority caluclation
          lamda              (float): coefficient for retrace operation
          num_arms             (int): number of multi arms
          burnin_length        (int): length of burnin to calculate qvalues
          unroll_length        (int): length of unroll to calculate qvalues
          target_update_period (int): how often to update the target parameters
        """

        
        self.env_name = env_name
        self.n_frames = n_frames
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space.n
        self.device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
        self.frame_process_func = get_preprocess_func(env_name)

        # define network
        self.in_online_q_network = QNetwork(self.action_space, n_frames)
        self.in_target_q_network = QNetwork(self.action_space, n_frames)
        self.ex_online_q_network = QNetwork(self.action_space, n_frames)
        self.ex_target_q_network = QNetwork(self.action_space, n_frames)
        
        self.embedding_net = EmbeddingNet(n_frames)
        self.embedding_classifier = EmbeddingClassifer(self.action_space)
        
        self.original_lifelong_net = LifeLongNet(n_frames)
        self.trained_lifelong_net = LifeLongNet(n_frames)
        
        # set optimizer
        self.in_q_optimizer = optim.Adam(self.in_online_q_network.parameters(), lr=in_q_lr)
        self.ex_q_optimizer = optim.Adam(self.ex_online_q_network.parameters(), lr=ex_q_lr)
        self.embedding_optimizer = optim.Adam(self.embedding_net.parameters(), lr=embed_lr)
        self.lifelong_optimizer = optim.Adam(self.trained_lifelong_net.parameters(), lr=lifelong_lr)

        self.in_q_clip_grad = in_q_clip_grad
        self.ex_q_clip_grad = ex_q_clip_grad
        self.embed_clip_grad = embed_clip_grad
        self.lifelong_clip_grad = lifelong_clip_grad
        
        self.criterion = torch.nn.CrossEntropyLoss()

        self.betas = create_beta_list(num_arms)
        self.gammas = create_gamma_list(num_arms)
        self.eta = eta
        self.lamda = lamda

        self.burnin_len = burnin_length
        self.unroll_len = unroll_length
        self.target_update_period = target_update_period

        self.num_updated = 0
    
    
    def set_device(self):
        """
        set network on device
        """
              
        self.in_online_q_network.to(self.device)
        self.in_target_q_network.to(self.device)
        self.ex_online_q_network.to(self.device)
        self.ex_target_q_network.to(self.device)
        self.embedding_net.to(self.device)
        self.embedding_classifier.to(self.device)
        self.trained_lifelong_net.to(self.device)
        self.original_lifelong_net.to(self.device)

    def define_network(self):
        """
        define network and get initial parameter to copy to angents
        """

        frame = self.frame_process_func(self.env.reset())
        frames = [frame] * self.n_frames
        
        # (1, n_frams, 32, 32)
        state = torch.tensor(np.stack(frames, axis=0)[None, ...]).float()
        h = torch.zeros(1, 1, self.in_online_q_network.lstm.hidden_size).float()
        c = torch.zeros(1, 1, self.ex_online_q_network.lstm.hidden_size).float()

        self.in_online_q_network(state,
                                 states=(h, c),
                                 prev_action=torch.tensor([0]),
                                 j=torch.tensor([0]),
                                 prev_in_rewards=torch.tensor([0]),
                                 prev_ex_rewards=torch.tensor([0]))
        self.ex_online_q_network(state,
                                 states=(h, c),
                                 prev_action=torch.tensor([0]),
                                 j=torch.tensor([0]),
                                 prev_in_rewards=torch.tensor([0]),
                                 prev_ex_rewards=torch.tensor([0]))

        self.in_target_q_network(state,
                                 states=(h, c),
                                 prev_action=torch.tensor([0]),
                                 j=torch.tensor([0]),
                                 prev_in_rewards=torch.tensor([0]),
                                 prev_ex_rewards=torch.tensor([0]))
        self.ex_target_q_network(state,
                                 states=(h, c),
                                 prev_action=torch.tensor([0]),
                                 j=torch.tensor([0]),
                                 prev_in_rewards=torch.tensor([0]),
                                 prev_ex_rewards=torch.tensor([0]))

        control_state = self.embedding_net(state)
        self.original_lifelong_net(state)
        self.trained_lifelong_net(state)
        self.embedding_classifier(control_state, control_state)

        self.in_target_q_network.load_state_dict(self.in_online_q_network.state_dict())
        self.ex_target_q_network.load_state_dict(self.ex_online_q_network.state_dict())

        in_q_weight = self.in_online_q_network.state_dict()
        ex_q_weight = self.ex_online_q_network.state_dict()
        embed_weight = self.embedding_net.state_dict()
        trained_lifelong_weight = self.trained_lifelong_net.state_dict()
        original_lifelong_weight = self.original_lifelong_net.state_dict()
        
        self.set_device()

        return in_q_weight, ex_q_weight, embed_weight, trained_lifelong_weight, original_lifelong_weight

    def save(self, weight_dir, cycle):
        """
        save weight
        Args:
          weight_dir (str): path to weight directory
          cycle      (int): the number of times the learning has been completed
        """

        torch.save(self.online_q_network.state_dict(), os.path.join(weight_dir, f"q_weight_{cycle}.pth"))
        torch.save(self.online_policy_net.state_dict(), os.path.join(weight_dir, f"policy_weight_{cycle}.pth"))
        torch.save(self.embedding_net.state_dict(), os.path.join(weight_dir, f"embed_weight_{cycle}.pth"))
        torch.save(self.embedding_classifier.state_dict(), os.path.join(weight_dir, f"embed_classifier_weight_{cycle}.pth"))
        torch.save(self.trained_lifelong_net.state_dict(), os.path.join(weight_dir, f"trained_lifelong_weight_{cycle}.pth"))
        torch.save(self.original_lifelong_net.state_dict(), os.path.join(weight_dir, f"original_lifelong_weight_{cycle}.pth"))

    @staticmethod
    def decompress_segments(minibatch):
        """
        decompress minibatch to indices, weights, segments
        Args:
          minibatch: minibatch of indices, weights and segments
        Returns:
          indices : indices of experiences
          weights : priorities of experiences
          segments: a coherent body of experience of some length
        """

        indices, weights, compressed_segments = minibatch
        segments = [pickle.loads(lz4f.decompress(compressed_seg))
                    for compressed_seg in compressed_segments]
        return indices, weights, segments

    def update_network(self, minibatchs):
        """
        update parameter of networks, generating losses.
        Args:
          minibatch: minibatch of indices, weights and segments
        Returns:
          weight and loss
        """

        indices_all = []
        priorities_all = []
        in_q_losses = []
        ex_q_losses = []
        embed_losses = []
        lifelong_losses = []

        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            work_in_progresses = [executor.submit(self.decompress_segments, minibatch) for minibatch in minibatchs]

            for ready_minibatch in futures.as_completed(work_in_progresses):
                indices, weights, segments = ready_minibatch.result()
                weights = torch.sqrt(torch.tensor(weights, requires_grad=True).float()).to(self.device)
                
                self.in_online_q_network.eval()
                self.in_target_q_network.eval()
                self.ex_online_q_network.eval()
                self.ex_target_q_network.eval()
                self.embedding_net.eval()
                self.trained_lifelong_net.eval()
                self.original_lifelong_net.eval()

                priorities, in_q_loss, ex_q_loss = self.qnet_update(weights, segments)
                embed_loss, lifelong_loss = self.ngu_update()

                indices_all += indices
                priorities_all += priorities.cpu().detach().numpy().tolist()
                in_q_losses.append(in_q_loss.cpu().detach().numpy())
                ex_q_losses.append(ex_q_loss.cpu().detach().numpy())
                embed_losses.append(embed_loss)
                lifelong_losses.append(lifelong_loss)

        in_q_weight = self.in_online_q_network.to('cpu').state_dict()
        ex_q_weight = self.ex_online_q_network.to('cpu').state_dict()
        embed_weight = self.embedding_net.to('cpu').state_dict()
        lifelong_weight = self.trained_lifelong_net.to('cpu').state_dict()
        
        self.set_device()

        return in_q_weight, ex_q_weight, embed_weight, lifelong_weight, indices_all, priorities_all, \
                np.mean(in_q_losses), np.mean(ex_q_losses), np.mean(embed_losses), np.mean(lifelong_losses)

    def qnet_update(self, weights, segments):
        """
        update q network
        Args:
          weights : priorities of experiences
          segments: a coherent body of experience of some length
        """

        self.states, self.actions, self.in_rewards, self.ex_rewards, self.dones, self.j, self.next_states, in_h0, in_c0, ex_h0, ex_c0, \
            self.prev_in_rewards, self.prev_ex_rewards, self.prev_actions = segments2contents(segments, burnin_len=self.burnin_len, is_grad=True, device=self.device)

        self.in_online_q_network.train()
        self.in_target_q_network.train()
        self.ex_online_q_network.train()
        self.ex_target_q_network.train()

        # (unroll_len+1, batch_size, action_space)
        in_online_qvalues = self.get_qvalues(self.in_online_q_network, in_h0, in_c0)

        # (unroll_len+1, batch_size, action_space)
        ex_online_qvalues = self.get_qvalues(self.ex_online_q_network, ex_h0, ex_c0)
        
        # (unroll_len+1, batch_size, action_space)
        in_target_qvalues = self.get_qvalues(self.in_target_q_network, in_h0, in_c0)

        # (unroll_len+1, batch_size, action_space)
        ex_target_qvalues = self.get_qvalues(self.ex_target_q_network, ex_h0, ex_c0)
        
        self.set_pi(ex_online_qvalues, in_online_qvalues)
        
        # (unroll_len, batch_size, action_space)
        self.actions_onehot = F.one_hot(self.actions[self.burnin_len:], num_classes=self.action_space)

        # (unroll_len, batch_size)
        in_online_Q = torch.sum(in_online_qvalues[:-1] * self.actions_onehot, dim=2)
        
        # (unroll_len, batch_size)
        ex_online_Q = torch.sum(ex_online_qvalues[:-1] * self.actions_onehot, dim=2)
        
        # (1, batch_size)
        self.gamma = torch.stack([self.gammas[i] for i in self.j], dim=0).unsqueeze(0).to(self.device)

        in_Retraced_Q = self.get_retraced_Q(in_target_qvalues, self.in_rewards)
        ex_Retraced_Q = self.get_retraced_Q(ex_target_qvalues, self.ex_rewards)

        self.in_q_optimizer.zero_grad()
        
        in_q_loss = F.mse_loss(weights*in_online_Q, weights*in_Retraced_Q)
        in_q_loss.backward(retain_graph=True)
        
        clip_grad_norm_(self.in_online_q_network.parameters(), self.in_q_clip_grad)
        self.in_q_optimizer.step()

        self.ex_q_optimizer.zero_grad()
        
        ex_q_loss = F.mse_loss(weights*ex_online_Q, weights*ex_Retraced_Q)
        ex_q_loss.backward(retain_graph=True)
        
        clip_grad_norm_(self.ex_online_q_network.parameters(), self.ex_q_clip_grad)
        self.ex_q_optimizer.step()

        in_td_errors = in_Retraced_Q - in_online_Q
        ex_td_errors = ex_Retraced_Q - ex_online_Q

        in_priorities = self.eta * torch.max(torch.abs(in_td_errors), dim=0).values + (1 - self.eta) * torch.mean(torch.abs(in_td_errors), dim=0)
        ex_priorities = self.eta * torch.max(torch.abs(ex_td_errors), dim=0).values + (1 - self.eta) * torch.mean(torch.abs(ex_td_errors), dim=0)
        priorities = in_priorities + ex_priorities

        # copy online q network parameter to target q network
        self.num_updated += 1
        if self.num_updated % self.target_update_period == 0:
            self.in_target_q_network.load_state_dict(self.in_online_q_network.state_dict())
            self.ex_target_q_network.load_state_dict(self.ex_online_q_network.state_dict())
            
        return priorities, in_q_loss, ex_q_loss

    def get_qvalues(self, q_network, h, c):
        """
        get qvalues from expeiences using specific q network
        Args:
          q_network             : network to get Q values
          h       (torch.tensor): LSTM hidden state
          c       (torch.tensor): LSTM cell state
        Returns:
          qvalues (torch.tensor): Q values [unroll_len+1, batch_size, action_space]
        """
        
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


    def set_pi(self, ex_online_qvalues, in_online_qvalues):
        """
        set pi from argmax of online qvalues 
        Args:
          ex_online_qvalues (torch.tensor): extrinsic Q values from online Q network [unroll_len+1, batch_size, action_space]
          in_online_qvalues (torch.tensor): intrinsic Q values from online Q network [unroll_len+1, batch_size, action_space]
        """
        
        # (1, batch_size, 1)
        beta = torch.stack([self.betas[i] for i in self.j], dim=0)[None, :, None].to(self.device)
        
        # (unroll_len+1, batch_size, action_space)
        online_qvalues = rescaling(inverse_rescaling(ex_online_qvalues) + beta * inverse_rescaling(in_online_qvalues))

        # (unroll_len+1, batch_size)
        self.pi = torch.argmax(online_qvalues, dim=2)
        
        # (unroll_len, batch_size, action_space)
        self.pi_onehot = F.one_hot(self.pi[1:], num_classes=self.action_space)
        

    def get_retraced_Q(self, target_qvalues, rewards):
        """
        implement retrace operation
        Args:
          target_qvalues (torch.tensor): Q values from target Q network [unroll_len+1, batch_size, action_space]
          rewards        (torch.tensor): rewards from experiences [burnin_len+unroll_len, batch_size]
        Returns:
          Retraced_Q (torch.tensor): Q values after retrace operation [unroll_len, batch_size]
        """
        
        # (unroll_len, batch_size)
        target_Q = torch.sum(target_qvalues[1:] * self.pi_onehot, dim=2)
        
        # (unroll_len, batch_size)
        Q = torch.sum(target_qvalues[:-1] * self.actions_onehot, dim=2)

        # (unroll_len, batch_size)
        TQ = rewards[self.burnin_len:] + self.gamma * (1 - self.dones) * inverse_rescaling(target_Q)

        # (unroll_len, batch_size)
        delta = TQ - inverse_rescaling(Q)

        # (unroll_len, batch_size)
        P = transformed_retrace_operator(delta=delta,
                                         pi=self.pi[:-1],
                                         actions=self.actions[self.burnin_len:],
                                         lamda=self.lamda,
                                         gamma=self.gamma.squeeze(0),
                                         unroll_len=self.unroll_len,
                                         device=self.device)

        Retraced_Q = rescaling(inverse_rescaling(Q) + P)

        return Retraced_Q

    def ngu_update(self):
        """
        update embedding network and lifelong network
        Returns:
          np.mean(embed_loss)    (np.ndarray): average of embedding loss
          np.mean(lifelong_loss) (np.ndarray): average of lifelong loss
        """

        embed_loss = []
        self.embedding_net.train()
        
        for t in range(self.burnin_len + self.unroll_len):
            
            control_state = self.embedding_net(self.states[t])
            next_control_state = self.embedding_net(self.next_states[t])
            action_prob = self.embedding_classifier(control_state, next_control_state)
            
            loss = self.criterion(action_prob, self.actions[t])
            self.embedding_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            clip_grad_norm_(self.embedding_net.parameters(), self.embed_clip_grad)
            self.embedding_optimizer.step()
            
            embed_loss.append(loss.cpu().detach().numpy())

        lifelong_loss = []
        self.trained_lifelong_net.train()
        
        for t in range(self.burnin_len + self.unroll_len):
            trained_output = self.trained_lifelong_net(self.states[t])
            original_output = self.original_lifelong_net(self.states[t])
            
            loss = F.mse_loss(trained_output, original_output)
            with open(f"log/lifelong_loss.txt", mode="a") as f:
                f.write(f"Cycle: {self.num_updated}, Loss: {loss}\n")
            self.lifelong_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            clip_grad_norm_(self.trained_lifelong_net.parameters(), self.lifelong_clip_grad)
            self.lifelong_optimizer.step()
            lifelong_loss.append(loss.cpu().detach().numpy())

        return np.mean(embed_loss), np.mean(lifelong_loss)
