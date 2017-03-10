# Import necessary libraries 
import numpy as np
import matplotlib.pyplot as plt

''' Define a Bandit class which has the following parameters
@param numArm : number of actions in the K-armed bandit problem
@param epsilon : Set the epsilon value for epsilon-greedy action selection. Default value is 0.0
@param temperature : Set the temperature parameter of softmax action selection. Default value is 1.0
@param epsilonGreedy : Flag to select that action-selection is based on epsilon-greedy algorithm. Default value is False
@param softmax : Flag to select that action-selection is based on the softmax algorithm. Defalut value is False
@param ucb : Flag to select that action-selection is based on the softmax algorithm. Default value is False
@param initial_estimate : Setting the initial estimates for estimates all the arms
@param trueReward : Setting the true reward for the all arms
'''

class Bandit:
    # Constructor function to initialise the parameters for the bandit instantiation
    def __init__(self, numArm = 10, epsilon = 0.0, temperature = 1.0, epsilonGreedy = False, softmax = False, ucb =
    False, initial_estimate = 0.0, trueReward = 0.0):
        # Initialisation of parameters
        self.numArm = numArm 
        self.epsilon = epsilon
        self.temperature = temperature
        self.time = 0
        self.averageReward = 0
        self.epsilonGreedy = epsilonGreedy
        self.softmax = softmax
        self.ucb = ucb
        self.armIndices = np.arange(self.numArm)
        self.qTrue = trueReward * np.ones(self.numArm) # True rewards for each action
        self.qEst = initial_estimate *  np.ones(self.numArm) # Estimate of the reward for each action
        self.actionCount = np.zeros(self.numArm) # Action Selection count
        # Get the true rewards for each action
        for i in range(0,self.numArm):
            self.qTrue[i] = self.qTrue[i] + np.random.randn()   
        self.bestAction = np.argmax(self.qTrue) 

    # Function to choose the action based on what algorithm is being followed
    def chooseAction(self):
        if self.epsilonGreedy == True:
            armToPlay = np.random.randint(0,self.numArm) # Choose a random action
            # With probability greater than epsilon, choose the action greedily
            if np.random.random() > self.epsilon:
                armToPlay = np.argmax(self.qEst)
            return armToPlay

        if self.softmax == True:
            expo_val = np.exp(self.qEst / self.temperature)
            prob_val = expo_val / np.sum(expo_val) # Get the probability distribution to be sampled from
            rnd = np.random.random() # Sample a random number
            # Use the CDF method to sample from the Gibbs distribution
            for i,w in enumerate(prob_val):
                rnd -= w
                if rnd < 0:
                    return i
            return 0

        if self.ucb == True:
            # play each action once at first
            if(np.count_nonzero(self.actionCount) < self.numArm):
                return np.argmin(self.actionCount)
            # after having played each arm once, choose action greedily using the upper confidence bound
            return np.argmax(self.qEst + np.sqrt(np.log(self.time + 1) / (np.asarray(self.actionCount) + 1)))

    # Function to get the reward on playing the action, increment the count of action and update the estimate of that action
    def getReward(self,action):
        reward = np.random.randn() + self.qTrue[action]
        self.time += 1
        self.averageReward = (self.time - 1.0) / self.time * self.averageReward + reward / self.time # keep a running average of the reward
        self.actionCount[action] += 1 # increase the action count
        self.qEst[action] += 1.0 / self.actionCount[action] * (reward - self.qEst[action]) # update the estimate of the action
        return reward