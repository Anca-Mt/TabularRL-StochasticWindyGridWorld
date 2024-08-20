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

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        # TO DO: Add own code
        target_Gt = r + (0 if done else self.gamma * self.Q_sa[s_next, a_next])
        self.Q_sa[s, a] += self.learning_rate * (target_Gt - self.Q_sa[s, a])

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    s = env.reset()
    a = agent.select_action(s, policy, epsilon, temp)
    total_reward = 0
    for timestep in range(1, n_timesteps + 1):
        s_next, r, done = env.step(a)
        a_next = agent.select_action(s_next, policy, epsilon, temp)
        agent.update(s, a, r, s_next, a_next, done)
        total_reward += r
        if done:
            s = env.reset()
            a= agent.select_action(s, policy, epsilon, temp)
        else:
            s = s_next
            a = a_next

        # Perform evaluation
        if timestep % eval_interval == 0:
            mean_return = agent.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(timestep)
            if plot:
                env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1)  # Plot the Q-value estimates during Q-learning execution

    print("Total reward: ", total_reward)
    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    eval_returns, eval_timesteps = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print(eval_returns, eval_timesteps)
    
if __name__ == '__main__':
    test()
