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

class QLearningAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,done):
        best_next_action = np.argmax(self.Q_sa[s_next])
        target_Gt = r + (0 if done else self.gamma * self.Q_sa[s_next, best_next_action])
        self.Q_sa[s,a] += self.learning_rate * (target_Gt - self.Q_sa[s,a])


def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=True)
    eval_env = StochasticWindyGridworld(initialize_model=True)
    '''env -> the environment used for training 
       eval_env -> a separate environment used for evaluation purposes
    '''
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your Q-learning algorithm here!
    s = env.reset()
    total_reward = 0
    for timestep in range(1, n_timesteps + 1):
        a = agent.select_action(s, policy, epsilon, temp)
        s_next, r, done = env.step(a)
        agent.update(s, a, r, s_next, done)

        total_reward += r
        if done:
            s = env.reset()
        else:
            s = s_next

        # Perform evaluation
        if timestep % eval_interval == 0:
            mean_return = agent.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(timestep)
            if plot:
                env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.1)  # Plot the Q-value estimates during Q-learning execution

    print("Total reward is: ", total_reward)

    return np.array(eval_returns), np.array(eval_timesteps)   

def test():
    
    n_timesteps = 1000
    eval_interval=100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'softmax' # 'egreedy' or 'softmax'
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
