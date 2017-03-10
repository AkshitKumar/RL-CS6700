''' Run this python script to generate the plots for the epsilon-greedy plot on a N-armed bandit testbed'''
from bandit import Bandit # import the bandit class to instantiate bandit objects for different bandit problems
from simulateBandit import simulateBandit # import the simulateBandit function to do the simulation of arm pulls and getting rewards
import matplotlib.pyplot as plt # for plotting graphs

''' Function used for plotting the average reward and optimal action selection using epsilon greedy action selection methods
@param numDiffBandits : number of different bandit problems
@param timeSteps : number of different time steps over which the algorithm is run
'''
def epsilonGreedy(numDiffBandits, timeSteps):
    epsilons = [0.0 , 0.1, 0.01] # taking three different values of epsilon
    bandits = list() # making a list of bandit problems
    # iterating through the list of epsilons and adding different bandit problems
    for epsilonIndex, epsilon in enumerate(epsilons):
        bandits.append([Bandit(epsilon = epsilon, epsilonGreedy = True) for _ in range(0,numDiffBandits)])
    optimalActionCount, averageReward = simulateBandit(numDiffBandits, timeSteps, bandits)
    # plotting the results for optimal action 
    plt.figure(0)
    for epsilon, counts in zip(epsilons, optimalActionCount):
        plt.plot(counts, label='epsilon = ' + str(epsilon))
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend(loc='best')
    # plotting the results for average reward
    plt.figure(1)
    for epsilon, rewards in zip(epsilons, averageReward):
        plt.plot(rewards, label='epsilon = ' + str(epsilon))
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend(loc='best')

epsilonGreedy(2000,1000) # running the simulation on 10-armed bandit testbed for 2000 different bandit problems over 1000 time steps
plt.show()
