"""
Reward functions for DQN RL

Written by Sami Saliba
"""

import numpy as np
import math

def reward_new(depth,flood):
    if(sum(flood) > 0):
        weights = np.full((3, 1), 0.5)
        x = 0
        iter = 0
        for i in flood:
            weights[x] *= iter
            iter += 1
            weights[x] += 1 / (1 + math.exp(-i))
            weights[x] = weights[x]/iter
            x += 1

        flood = [-(i**2) if i > 0.0 else 0 for i in flood]
        reward = np.dot(flood, weights)

    else:
        depths = [math.fabs(i - 2) for i in depth]
        weights = np.full((3, 1), -0.75)
        reward = np.dot(depths, weights)
    return reward[0]


def reward_function2(depth, flood):
    flood = [-((i)) if i > 0.0 else 0 for i in flood]
    weights = [1, 1, 1]
    flood_reward = np.dot(flood, np.transpose(weights))
    # Sum the total reward
    total_reward = flood_reward
    return total_reward

def reward_function3(depth, flood):
    flood = [-(i) if i > 0.0 else 0 for i in flood]
    weights = [.5, .5, .5, 2, 2]
    flood_reward = np.dot(flood, np.transpose(weights))
    # Sum the total reward
    total_reward = flood_reward
    return total_reward


def reward_function1(depth, flood):
    weights = np.full((5, 1), 0.5)
    x = 0
    iter = 0
    for i in flood:
        weights[x] *= iter
        iter += 1
        weights[x] += 1 / (1 + math.exp(-i))
        weights[x] = weights[x]/iter
        x += 1

    flood = [-(i**2) if i > 0.0 else 0 for i in flood]
    flood_reward = np.dot(flood, weights)

    return flood_reward[0]


def reward_function4(depth, flood):
    weights = np.full((5, 1), 0.5)
    x = 0
    iter = 0
    for i in flood:
        weights[x] *= iter
        iter += 1
        weights[x] += 1 / (1 + math.exp(-i))
        weights[x] = weights[x]/iter
        x += 1

    flood = [-(i) if i > 0.0 else 0 for i in flood]
    flood_reward = np.dot(flood, weights)

    return flood_reward[0]


def reward_function5(depth, flood):
    flood = [-(i**2) if i > 0.0 else 0 for i in flood]
    weights = [1, 1, 1, 1, 1]
    flood_reward = np.dot(flood, np.transpose(weights))
    # Sum the total reward
    total_reward = flood_reward
    return total_reward

def reward_function10(depth, flood):
    flood = [-((i)**2) if i > 0.0 else 0 for i in flood]
    weights = [.5, .5, .5, 2, 2]
    flood_reward = np.dot(flood, np.transpose(weights))
    # Sum the total reward
    total_reward = flood_reward
    return total_reward

def reward_function6(depth, flood):
    weights = np.full((5, 1), 0.5)
    x = 0
    iter = 0
    for i in flood:
        weights[x] *= iter
        iter += 1
        weights[x] = weights[x]/iter
        x += 1

    flood = [-(i) if i > 0.0 else 0 for i in flood]
    flood_reward = np.dot(flood, weights)

    return flood_reward[0]

def reward_function7(depth, flood):
    weights = np.full((5, 1), 0.5)
    x = 0
    iter = 0
    for i in flood:
        weights[x] *= iter
        iter += 1
        weights[x] = weights[x]/iter
        x += 1

    flood = [-(i)**2 if i > 0.0 else 0 for i in flood]
    flood_reward = np.dot(flood, weights)

    return flood_reward[0]


    
def reward_function8(depth, flood):
    weights = np.full((5, 1), 0.5)
    for i in flood:
        i += 1 / (1 + math.exp(-i))

    flood = [-(i) if i > 0.0 else 0 for i in flood]
    flood_reward = np.dot(flood, weights)

    return flood_reward[0]


def reward_function9(depth, flood):
    weights = np.full((5, 1), 0.5)
    for i in flood:
        i += 1 / (1 + math.exp(-i))

    flood = [-(i)**2 if i > 0.0 else 0 for i in flood]
    flood_reward = np.dot(flood, weights)

    return flood_reward[0]

'''def reward_function8(depth, flood):
    weights = np.full((5, 1), 0.5)

    for ii in weights:
        ii = 20 * math.log(ii+1, 10)

    flood = [-(i) if i > 0.0 else 0 for i in flood]
    flood_reward = np.dot(flood, weights)

    return flood_reward[0]

def reward_function9(depth, flood):
    weights = np.full((5, 1), 0.5)

    for ii in weights:
        ii = 20 * math.log(ii+1, 10)

    flood = [-(i)**2 if i > 0.0 else 0 for i in flood]
    flood_reward = np.dot(flood, weights)

    return flood_reward[0]
'''