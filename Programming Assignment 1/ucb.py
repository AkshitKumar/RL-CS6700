''' Run this python script to generate the plots for comparing the UCB algorithm vs epsilon greedy and softmax on a N-armed bandit testbed'''
from bandit import Bandit # import the bandit class to instantiate bandit objects for different bandit problems
from simulateBandit import simulateBandit # import the simulateBandit function to do the simulation of arm pulls and getting rewards
import matplotlib.pyplot as plt # for plotting graphs

''' Function used for plotting the average reward and optimal action selection using ucb,epsilon greedy and softmax action selection methods
@param numArms : number of arms in the testbed
@param numDiffBandits : number of different bandit problems
@param timeSteps : number of different time steps over which the algorithm is run
'''
def ucb(numArms,numDiffBandits, timeSteps):
    algos = ['UCB','Softmax','Epsilon Greedy'] # iterating through the different algorithms
    bandits = [[],[],[]]
    bandits[0] = [Bandit(numArm = numArms, ucb = True) for _ in range(0,numDiffBandits)] # list of bandits following ucb
    bandits[1] = [Bandit(numArm = numArms, softmax = True, temperature = 0.1) for _ in range(0,numDiffBandits)] # list of bandits following softmax action selection
    bandits[2] = [Bandit(numArm = numArms, epsilonGreedy = True, epsilon = 0.1) for _ in range(0,numDiffBandits)] # list of bandits following epsilon greedy selection 
    optimalActionCount, averageReward = simulateBandit(numDiffBandits, timeSteps, bandits)
    # plotting the results for optimal action 
    plt.figure(0)
    plt.plot(optimalActionCount[0],label='UCB')
    plt.plot(optimalActionCount[1],label='Softmax')
    plt.plot(optimalActionCount[2],label='Epsilon-Greedy')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend(loc = 'best')
    # plotting the results for average reward
    plt.figure(1)
    plt.plot(averageReward[0],label = 'UCB')
    plt.plot(averageReward[1],label = 'Softmax')
    plt.plot(averageReward[2],label = 'Epsilon-Greedy')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend(loc = 'best')

ucb(1000,2000,20000) # running the simulation on 1000-armed bandit testbed for 2000 different bandit problems over 20000 time steps
plt.show()
