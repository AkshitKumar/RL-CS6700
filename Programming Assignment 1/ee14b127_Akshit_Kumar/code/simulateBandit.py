# Import the necessary libraries
import numpy as np

''' Function to simulate the bandit problem
@param numDiffBandits : variable stating the number of different bandit problems
@param timeSteps : variable stating the number of time steps over which the actions are taken
@param bandits : list contains the different bandit problems
'''
def simulateBandit(numDiffBandits, timeSteps, bandits):
    optimalActionCount = np.zeros(shape=(len(bandits),timeSteps)) # 2D matrices for storing the count of optimal action selection in each time step for all the different bandit problems
    averageReward = np.zeros(shape=(len(bandits),timeSteps)) # 2D matrices for storing the average reward in each time step for all the different bandit problems
    for banditIndex, bandit in enumerate(bandits): # iterating through the all the different bandit problems
        for i in range(0,numDiffBandits): # iterating through the number of different bandits
            for j in range(0,timeSteps): # iterating through the time steps for each bandit problem
                action = bandit[i].chooseAction() # select an action to take at each time step depending on the action selection method for the particular bandit
                reward = bandit[i].getReward(action) # get a reward for selecting that action for that particular time step
                averageReward[banditIndex][j] += reward # calculate the cummulative reward
                if action == bandit[i].bestAction:
                    optimalActionCount[banditIndex][j] += 1 # increment the count of action by 1
        optimalActionCount[banditIndex] /= numDiffBandits # average out the optimal action count
        averageReward[banditIndex] /= numDiffBandits # average out the cummulative reward
    return optimalActionCount, averageReward


