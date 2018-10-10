import gym
import numpy as np

env = gym.make('Taxi-v2')
num_states = env.observation_space.n
num_actions = env.action_space.n
max_iterations = 1000
delta = 10**-3

R = np.zeros([num_states, num_actions, num_states])
T = np.zeros([num_states, num_actions, num_states])
V = np.zeros([env.observation_space.n])
Q = np.zeros([env.observation_space.n, env.action_space.n])
gamma = 0.9

print("Taxi-v2")
print("Actions: ", num_actions)
print("States: ", num_states)
print(env.env.desc)

for state in range(num_states):
    for action in range(num_actions):
        for transition in env.env.P[state][action]:
            probability, next_state, reward, done = transition
            R[state, action, next_state] = reward
            T[state, action, next_state] = probability
        T[state, action, :] /= np.sum(T[state, action, :])

value_fn = np.zeros([num_states])

for i in range(max_iterations):
    previous_value_fn = value_fn.copy()
    #learn more about einsum
    Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
    value_fn = np.max(Q, axis=1)
    if np.max(np.abs(value_fn - previous_value_fn)) < delta:
            break
    policy = np.argmax(Q, axis=1)
iters = i + 1

print("Value Iteration")
print("Iterations: ", iters)
