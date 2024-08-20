#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class NstepQLearningAgent(BaseAgent):
        
    def update(self, states, actions, rewards, done, n):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        T_ep = len(states)-1   # Terminal time step
        for t in range(T_ep):
            m = min(n, T_ep - t)
            target_Gt = sum([self.gamma ** i * rewards[t+i] for i in range(m)])
            if t + m < T_ep:
                target_Gt += self.gamma ** m * np.max(self.Q_sa[states[t+m]])
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (target_Gt - self.Q_sa[states[t], actions[t]])

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your n-step Q-learning algorithm here!
    total_reward = 0
    timestep = 0
    while timestep < n_timesteps:
        s = env.reset()
        states, actions, rewards = [], [], []
        for i in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_next
            total_reward += r
            timestep +=1
            if done:
                s = env.reset()

        pi.update(states, actions, rewards, done,  n)

        if timestep % eval_interval == 0:
            eval_return = pi.evaluate(eval_env)
            eval_returns.append(eval_return)
            eval_timesteps.append(timestep)
            if plot:
                env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.2) # Plot the Q-value estimates during Monte Carlo RL execution

    print("Total reward is: ", total_reward)

    return np.array(eval_returns), np.array(eval_timesteps)

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    
    
if __name__ == '__main__':
    test()
