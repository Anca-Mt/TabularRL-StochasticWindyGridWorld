#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import argmax
import matplotlib.pyplot as plt

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        # TO DO: Add own code
        a = np.argmax(self.Q_sa[s])
        '''COMMENT FOR ME:  np.argmax finds the index of the maximum value in that row, which corresponds 
            to the action with the highest Q-value for that state.
            This code selects the action based on the highest Q value in state s
        '''
        # a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return a
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        Q_sa_previous = self.Q_sa[s, a]
        self.Q_sa[s,a] = np.sum(p_sas*(r_sas + self.gamma*np.max(self.Q_sa, axis=1)))
        error = np.abs(Q_sa_previous - self.Q_sa[s, a])
        return self.Q_sa[s,a], error
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

     # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))

    Q_sa = None
    max_abs_error = float('inf')
    iteration_count = 0

    Q_values_history = []

    while max_abs_error > threshold:
        max_abs_error = 0
        Q_values_iteration = np.zeros((env.n_states, env.n_actions))  # Initialize Q-values for this iteration
        for s in range(env.n_states):
            for a in range(env.n_actions):

                p_sas, r_sas = env.model(s, a)
                Q_sa, error = QIagent.update(s,a,p_sas,r_sas)
                Q_values_iteration[s, a] = Q_sa
                max_abs_error = max(max_abs_error, error)

        # Store the Q-values for this iteration
        Q_values_history.append(Q_values_iteration)


        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        print("Q-value iteration, iteration {}, max error {}".format(iteration_count, max_abs_error))
        iteration_count += 1


    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent= Q_value_iteration(env,gamma,threshold)

    V = np.max(QIagent.Q_sa, axis=1)
    # view optimal policy
    done = False
    s = env.reset()

    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
        s = s_next


    V_s3 = V[0] # equivalent with the first state
    print(f"V(s=3) at the start state: {V_s3}")

    # Other method to compute V_s3:
    # Q_opt_sa = QIagent.Q_sa[env.start_location]
    # V_opt_s3 = np.max(Q_opt_sa)
    # print(f"V(s=3) NEW at the start state: {V_opt_s3}")

    # TO DO: Compute mean reward per timestep under the optimal policy
    E = 10000
    normalized_gains = []
    for episode in range(E):
        done = False
        s = env.reset()
        g = 0  # Total return for the episode

        while not done:
            a = QIagent.select_action(s)
            s_next, r, done = env.step(a)
            g += r  # Accumulate the total reward
            s = s_next

        normalized_g = g / (100 - g)
        normalized_gains.append(normalized_g)

    average_normalized_return = sum(normalized_gains) / len(normalized_gains)
    print(f"Average normalized return over {E} episodes: {average_normalized_return}")


if __name__ == '__main__':
    experiment()
