import collections
import random

import gym
import numpy as np
import torch
from PIL import Image


def rescaling(x):
    """https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L180
    """
    
    eps = 0.001
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + eps * x


def inverse_rescaling(x):
    """https://github.com/google-research/seed_rl/blob/f53c5be4ea083783fb10bdf26f11c3a80974fa03/agents/r2d2/learner.py#L186
    """
    
    eps = 0.001
    return torch.sign(x) * (torch.square(((torch.sqrt(1. + 4. * eps * (torch.abs(x) + 1. + eps))) - 1.) / (2. * eps)) - 1.)


def get_preprocess_func(env_name):
    
    if "Breakout" in env_name:
        return _preprocess_breakout
    elif "Pacman" in env_name:
        return _preprocess_mspackman
    else:
        raise NotImplementedError(f"Frame processor not implemeted for {env_name}")


def _preprocess_breakout(frame, resize=84):
    
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 34, 160, 200)).resize((resize, resize))
    image_scaled = np.array(image) / 255.0
    return image_scaled.astype(np.float32)


def _preprocess_mspackman(frame, resize=84):
    
    image = Image.fromarray(frame)
    image = image.convert("L").crop((0, 0, 160, 170)).resize((resize, resize))
    image_scaled = np.array(image) / 255.0
    return image_scaled.astype(np.float32)


def get_initial_lives(env_name):
    if "Breakout" in env_name:
        return 5
    elif "Pacman" in env_name:
        return 3
    else:
        raise NotImplementedError(f"Frame processor not implemeted for {env_name}")


def seed_evrything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def create_beta_list(num_arms, beta=0.3):
    
    betas = [torch.tensor(0)]
    for i in range(1, num_arms-1):
        betas.append(beta * torch.sigmoid(torch.tensor(10 * (2*i / (num_arms-2) - 1))))
    betas.append(torch.tensor(beta))
    return betas


def create_gamma_list(num_arms, gamma0=0.9999, gamma1=0.997, gamma2=0.99):
    
    gammas = [torch.tensor(gamma0)]
    for i in range(1, 7):
        gammas.append(gamma0 + (gamma1 - gamma0) * torch.sigmoid(torch.tensor(10 * (i - 3) / 3)))
    gammas.append(torch.tensor(gamma1))
    
    for i in range(8, num_arms):
        t = (num_arms-i-1) * torch.log(torch.tensor(1-gamma1)) + (i-8) * torch.log(torch.tensor(1-gamma2))
        gammas.append(1 - torch.exp(t / (num_arms-9)))
    
    return gammas


class UCB:
    def __init__(self, num_arms, window_size, epsilon, beta):
        
        self.data = collections.deque(maxlen=window_size)
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.beta = beta
        self.count = 0

    def pull_index(self):
        
        if self.count < self.num_arms:
            index = self.count
            self.count += 1
            
        else:
            if random.random() > self.epsilon:
                N = np.zeros(self.num_arms)
                mu = np.zeros(self.num_arms)
                
                for j, reward in self.data:
                    N[j] += 1
                    mu[j] += reward
                mu = mu / (N + 1e-10)
                index = np.argmax(mu + self.beta * np.sqrt(1 / (N + 1e-6)))
                
            else:
                index = np.random.choice(self.num_arms)
        return index

    def push_data(self, datas):
        
        self.data += [(j, reward) for j, reward in datas]


def get_episodic_reward(x, M, k, c=0.001, epsilon=0.0001, cluster_distance=0.008, max_similarity=8):
    
    dist_list = [np.linalg.norm((m -x), ord=2) for m in M]

    topk_dist_list = np.sort(dist_list)[:k]
    dm = np.mean(topk_dist_list) 
    
    if dm == 0:
        return 1e-10
    else:
        topk_dist_list = topk_dist_list / dm
        topk_dist_list = np.where(topk_dist_list-cluster_distance<0, 0, topk_dist_list-cluster_distance)
    
    K = epsilon / (epsilon + topk_dist_list)
    s = np.sqrt(np.sum(K)) + c

    if s > max_similarity:
        return 1e-10
    else:
        return 1 / s


def transformed_retrace_operator(delta, pi, actions, gamma, unroll_len, lamda, device=torch.device("cpu")):
    
    # (unroll_len, batch_size)
    P_list = delta  
    
    # (unroll_len, batch_size)
    C = torch.where(pi == actions, torch.tensor(lamda).to(device), torch.tensor(0.).to(device))  
    
    for t in range(unroll_len-2, -1, -1):
        P_list[t, :] += gamma * C[t+1, :] * P_list[t+1, :]
        
    return P_list


def play_episode(frame_process_func,
                 env_name,
                 n_frames,
                 action_space,
                 j,
                 epsilon,
                 k,
                 error_list,
                 L,
                 in_q_network,
                 ex_q_network,
                 embedding_net,
                 original_lifelong_net,
                 trained_lifelong_net,
                 beta=0.3,
                 is_test=False):
    
    env = gym.make(env_name)
    frame = frame_process_func(env.reset())
    
    # (n_frames, 84, 84)
    frames = collections.deque([frame] * n_frames, maxlen=n_frames)

    in_h = ex_h = torch.zeros(1, 1, in_q_network.lstm.hidden_size).float()
    in_c = ex_c = torch.zeros(1, 1, in_q_network.lstm.hidden_size).float()
    prev_action = np.random.choice(action_space)
    prev_ex_reward = 0
    prev_in_reward = 0
    episode_reward = 0
    done = False

    lives = get_initial_lives(env_name)
    
    M = collections.deque(maxlen=int(1e3))
    ucb_datas = []
    transitions = []

    while not done:
        
        # batching (1, n_frames, 84, 84)
        state = torch.tensor(np.stack(frames, axis=0)[None, ...]).float()
        
        # intrinsic Qvalues (1, action_space)
        in_qvalue, (next_in_h, next_in_c) = in_q_network(state,
                                                         states=(in_h, in_c),
                                                         prev_action=torch.tensor([prev_action]),
                                                         j=torch.tensor([j]),
                                                         prev_ex_rewards=torch.tensor([prev_ex_reward]).float(),
                                                         prev_in_rewards=torch.tensor([prev_in_reward]).float())
        # extrinsic Qvalues (1, action_space)
        ex_qvalue, (next_ex_h, next_ex_c) = ex_q_network(state,
                                                         states=(ex_h, ex_c),
                                                         prev_action=torch.tensor([prev_action]),
                                                         j=torch.tensor([j]),
                                                         prev_ex_rewards=torch.tensor([prev_ex_reward]).float(),
                                                         prev_in_rewards=torch.tensor([prev_in_reward]).float())

        # ε-greedy
        if random.random() < epsilon:
            action = np.random.choice(action_space)
        else:
            # concat with rescaling
            qvalue = rescaling(inverse_rescaling(ex_qvalue) + beta * inverse_rescaling(in_qvalue))
            action = np.argmax(qvalue.detach().numpy())

        # step enviroment
        next_frame, ex_reward, done, info = env.step(action)
        frames.append(frame_process_func(next_frame))
        
        # batching (1, n_frames, 84, 84)
        next_state = np.stack(frames, axis=0)[None, ...]

        control_state = embedding_net(state).squeeze(0).detach().numpy()
        error = np.square(original_lifelong_net(state).detach().numpy(), trained_lifelong_net(state).detach().numpy()).mean()
        
        if len(M) < k: 
            episodic_reward = 0
            std = 1
            avg = 1
        else:
            episodic_reward = get_episodic_reward(control_state, M, k)
            std = np.std(error_list)
            avg = np.mean(error_list)
            
        curiosity = 1 + (error - avg) / (std + 1e-10)
        
        # push data to Memory
        M.append(control_state)        
        error_list.append(error)
        
        in_reward = episodic_reward * np.clip(curiosity, 1, L)

        if is_test:
            episode_reward += ex_reward
            
        else:
            if lives != info["ale.lives"] or done:  # done==True when lose life
                lives = info["ale.lives"]
                transition = (prev_ex_reward, prev_in_reward, prev_action,
                              state, action, in_h, in_c, ex_h, ex_c, j,
                              True, ex_reward, in_reward, next_state)
            else:
                transition = (prev_ex_reward, prev_in_reward, prev_action,
                              state, action, in_h, in_c, ex_h, ex_c, j,
                              done, ex_reward, in_reward, next_state)
            transitions.append(transition)

        ucb_datas.append((j, ex_reward))

        in_h, in_c, ex_h, ex_c = next_in_h, next_in_c, next_ex_h, next_ex_c
        prev_action, prev_ex_reward, prev_in_reward = action, ex_reward, in_reward

    if is_test:
        return ucb_datas, episode_reward, error_list
    else:
        return ucb_datas, transitions, error_list


def segments2contents(segments, burnin_len, is_grad=False, device=torch.device("cpu")):
    
    # (burnin_len+unroll_len, batch_size, n_frames, 84, 84)
    states = torch.stack([torch.tensor(np.vstack(seg.states), requires_grad=is_grad) for seg in segments], dim=1).float().to(device)
    
    # (burnin_len+unroll_len, batch_size)
    actions = torch.stack([torch.tensor(seg.actions) for seg in segments], dim=1).to(device)
    
    # (burnin_len+unroll_len, batch_size)
    ex_rewards = torch.stack([torch.tensor(seg.ex_rewards, requires_grad=is_grad) for seg in segments], dim=1).float().to(device)
    
    # (burnin_len+unroll_len, batch_size)
    in_rewards = torch.stack([torch.tensor(seg.in_rewards, requires_grad=is_grad) for seg in segments], dim=1).float().to(device)
    
    # (unroll_len, batch_size)
    dones = torch.stack([torch.tensor(seg.dones[burnin_len:]) for seg in segments], dim=1).float().to(device)

    # (batch_size,)
    j = torch.stack([torch.tensor(seg.j) for seg in segments], dim=0).to(device)
    
    # (batch_size, n_frames, 84, 84)
    last_state = torch.stack([torch.tensor(np.vstack(seg.last_state), requires_grad=is_grad) for seg in segments], dim=0).float().to(device)
    
    # (burnin_len+unroll_len, batch_size, n_frames, 84, 84)
    next_states = torch.cat([states, last_state[None, :]], dim=0)[1:].to(device)

    # (1, batch_size, hidden_size)
    in_h0 = torch.cat([seg.in_h_init for seg in segments], dim=1).float().to(device)
    
    # (1, batch_size, hidden_size)
    in_c0 = torch.cat([seg.in_c_init for seg in segments], dim=1).float().to(device)
    
    # (1, batch_size, hidden_size)
    ex_h0 = torch.cat([seg.ex_h_init for seg in segments], dim=1).float().to(device)
    
    # (1, batch_size, hidden_size)
    ex_c0 = torch.cat([seg.ex_c_init for seg in segments], dim=1).float().to(device)

    # (batch_size)
    in_reward0 = torch.tensor([seg.prev_in_reward_init for seg in segments]).float().to(device)
    
    # (batch_size)
    ex_reward0 = torch.tensor([seg.prev_ex_reward_init for seg in segments]).float().to(device)
    
    # (burnin+unroll_len, batch_size)
    prev_in_rewards = torch.cat([in_reward0[None, :], in_rewards], dim=0)[:-1]
    
    # (burnin+unroll_len, batch_size)
    prev_ex_rewards = torch.cat([ex_reward0[None, :], ex_rewards], dim=0)[:-1]

    # (batch_size)
    a0 = torch.tensor([seg.prev_a_init for seg in segments]).to(device)
    
    # (burnin+unroll_len, batch_size)
    prev_actions = torch.cat([a0[None, :], actions], dim=0)[:-1]

    return states, actions, ex_rewards, in_rewards, dones, j, next_states, in_h0, in_c0, ex_h0, ex_c0, prev_in_rewards, prev_ex_rewards, prev_actions
