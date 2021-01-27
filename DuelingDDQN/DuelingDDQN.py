import os
import torch as T
import torch.nn.functional as F
import numpy as np
from DuelingDDQN.replaybuffer import ReplayBuffer
from DuelingDDQN.network import DuelingDeepQNetwork

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, fc1_dims=512, eps_min=0.01, 
                eps_dec=5e-7, replace=1000, chkpt_dir='DuelingDDQN/tmp/dueling_ddqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions 
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayBuffer(mem_size, input_dims)
        self.batch_size = batch_size

        self.q_eval = DuelingDeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, 
                                fc1_dims=self.fc1_dims, name='dueling_ddqn_q_eval', chkpt_dir=self.chkpt_dir)

        self.q_next = DuelingDeepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, 
                                fc1_dims=self.fc1_dims, name='dueling_ddqn_q_next', chkpt_dir=self.chkpt_dir)

        self.learn_step_cntr = 0

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()

        else:
            action = np.random.choice(self.action_space)

        return action

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def replace_target_network(self):
        if self.learn_step_cntr % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return False

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_cntr += 1
        self.decrement_epsilon()

        return loss

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()






    
