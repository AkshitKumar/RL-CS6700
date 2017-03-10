''' Run this python script to generate the plots for the softmax action selection plots on a N-armed bandit testbed'''
from bandit import Bandit # import the bandit class to instantiate bandit objects for different bandit problems
from simulateBandit import simulateBandit # import the simulateBandit function to do the simulation of arm pulls and getting rewards
import matplotlib.pyplot as plt # for plotting graphs

''' Function used for plotting the average reward and optimal action selection using softmax action selection methods
@param numDiffBandits : number of different bandit problems
@param timeSteps : number of different time steps over which the algorithm is run
'''
def softmax(numDiffBandits,timeSteps):
    temperatures = [0.01,0.1,1,10]  # taking 4 different values of temperate for the softmax distribution
    bandits = list() # making a list of bandit problems
    # iterating through the list of temperatures and adding different bandit problems
    for tempIndex, temp in enumerate(temperatures):
        bandits.append([Bandit(temperature = temp, softmax = True) for _ in range(0,numDiffBandits)])
    optimalActionCount, averageReward = simulateBandit(numDiffBandits, timeSteps, bandits)
    # plotting the results for optimal action 
    plt.figure(0)
    for temp, counts in zip(temperatures, optimalActionCount):
        plt.plot(counts, label='temp = ' + str(temp))
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend(loc='best')
    # plotting the results for average reward
    plt.figure(1)
    for temp, rewards in zip(temperatures, averageReward):
        plt.plot(rewards, label='temp = ' + str(temp))
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend(loc='best')
    
softmax(2000,1000) # running the simulation on 10-armed bandit testbed for 2000 different bandit problems over 1000 time steps
plt.show()
