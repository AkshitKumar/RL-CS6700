import random
import csv
from copy import copy, deepcopy
import matplotlib.pyplot as plt

empty = 0
player_x = 1
player_o = 2
draw = 3

def emptyState():
    return [[empty,empty,empty],[empty,empty,empty],[empty,empty,empty]]

def gameOver(state):
    for i in range(3):
        if(state[i][0] != empty and state[i][0] == state[i][1] and state[i][0] == state[i][2]):
            return state[i][0]
        if(state[0][i] != empty and state[0][i] == state[1][i] and state[0][i] == state[2][i]):
            return state[0][i]
        
    if(state[0][0] != empty and state[0][0] == state[1][1] and state[0][0] == state[2][2]):
        return state[0][0]
    
    if(state[2][0] != empty and state[2][0] == state[1][1] and state[2][0] == state[0][2]):
        return state[2][0]
    
    for i in range(3):
        for j in range(3):
            if state[i][j] == empty:
                return empty
            
    return draw


def lastToAct(state):
    countx = 0
    counto = 0    
    for i in range(3):
        for j in range(3):
            if state[i][j] == player_x:
                countx += 1
            elif state[i][j] == player_o:
                counto += 1
    if countx == counto:
        return player_o
    if (countx == counto + 1):
        return player_x
    return -1


def enumStates(state,idx,agent):
    if idx > 8:
        player = lastToAct(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameOver(state)
        if(winner != empty):
            return
        i = idx/3
        j = idx % 3
        for val in range(3):
            state[i][j] = val
            enumStates(state,idx+1,agent)
            

class Agent(object):
    def __init__(self,player,verbose = False, lossval = 0, learning = True):
        self.values =  {}
        self.player = player
        self.verbose = verbose
        self.lossval = lossval
        self.learning = learning
        self.epsilon = 0.1
        self.alpha = 0.1
        self.prevState = None
        self.prevScore = 0
        self.count = 0
        enumStates(emptyState(),0,self)
        
    def episode_over(self,winner):
        self.backup(self.winnerval(winner))
        self.prevState = None
        self.prevScore = 0
        
    def action(self,state):
        r = random.random()
        if r < self.epsilon :
            move = self.random(state)
        else:
            move = self.greedy(state)
        state[move[0]][move[1]] = self.player
        self.prevState = self.stateTuple(state)
        self.prevScore = self.lookup(state)
        state[move[0]][move[1]] = empty
        return move
    
    def random(self,state):
        available = []
        for i in range(3):
            for j in range(3):
                if(state[i][j] == empty):
                    available.append((i,j))
        return random.choice(available)

    def greedy(self,state):
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == empty:
                    state[i][j] = self.player
                    val = self.lookup(state)
                    state[i][j] = empty
                    if val > maxval:
                        maxval = val
                        maxmove = (i,j)
                    if self.verbose:
                        cells.append('{0:.3f}'.format(val).center(6))
                elif self.verbose:
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print BOARD_FORMAT.format(*cells)
        self.backup(maxval)
        return maxmove
    
    def backup(self,nextval):
        if self.prevState != None and self.learning:
            self.values[self.prevState] += self.alpha * ( nextval - self.prevScore)
            
    def lookup(self,state):
        key= self.stateTuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]
    
    def add(self,state):
        winner = gameOver(state)
        tup = self.stateTuple(state)
        self.values[tup] = self.winnerval(winner)
        
    def stateTuple(self,state):
        return (tuple(state[0]),tuple(state[1]),tuple(state[2]))
    
    def winnerval(self,winner):
        if winner == self.player:
            return 1
        elif winner == empty:
            return 0.5
        elif winner == draw:
            return 0
        else:
            return self.lossval
    

def play(agent1,agent2):
    state = emptyState()
    for i in range(9):
        if i % 2 == 0:
            move = agent1.action(state)
        else:
            move = agent2.action(state)
        state[move[0]][move[1]] = (i%2) + 1
        winner = gameOver(state)
        if winner != empty:
            return winner
    return winner

def measure_performance_self_play(agent1,agent2):
    probs = [0,0,0]
    games = 100
    for i in range(games):
        winner = play(agent1,agent2)
        if winner == player_x:
            probs[0] += 1.0/games
        elif winner == player_o :
            probs[1] += 1.0/games
        else:
            probs[2] += 1.0/games
    return probs

def measure_performance_vs_random(agent1,agent2):
    epsilon1 = agent1.epsilon
    epsilon2 = agent2.epsilon
    agent1.epsilon = 0
    agent2.epsilon = 0
    agent1.learning = False
    agent2.learning = False
    r1 = Agent(1)
    r2 = Agent(2)
    r1.epsilon = 1
    r2.epsilon = 1
    probs = [0,0,0,0,0,0]
    games = 1000
    for i in range(games):
        winner = play(agent1,r2)
        if winner == player_x:
            probs[0] += 1.0/games
        elif winner == player_o:
            probs[1] += 1.0/games
        else:
            probs[2] += 1.0/games
    for i in range(games):
        winner = play(r1,agent2)
        if winner == player_x:
            probs[3] += 1.0/games
        elif winner == player_o:
            probs[4] += 1.0/games
        else:
            probs[5] += 1.0/games
    agent1.epsilon = epsilon1
    agent2.epsilon = epsilon2
    agent1.learning = True
    agent2.learning = True
    return probs

def measure_performance_vs_random_player(agent1,r1):
    epsilon = agent1.epsilon
    agent1.epsilon = 0
    agent1.learning = False
    random_player = Agent(2)
    random_player.epsilon = 1
    probs = [0,0,0]
    games = 1000
    for i in range(games):
        winner = play(agent1,random_player)
        if winner == player_x:
            probs[0] += 1.0/games
        elif winner == player_o:
            probs[1] += 1.0/games
        else:
            probs[2] += 1.0/games
    agent1.epsilon = epsilon
    agent1.learning = True
    return probs
    
if __name__ == "__main__":
    p1 = Agent(1,lossval = 0)
    #p1.epsilon = 0
    p2 = Agent(2,lossval = 0)
    #p2.epsilon = 0
    series = ['Agent-1 Wins','Agent-1 Loses','Agent-1 Draws','Agent-2 Wins','Agent-2 Loses','Agent-2 Draws']
    series_self_play = ['SPA Wins','SPA Loses','SPA Draws']
    colors_self_play = ['c','m','k']
    colors = ['r','b','g','c','m','b']
    markers = ['+','.','o','*','^','s']
    series_random = ['RTA Wins','RTA Loses','RTA Draws']
    colors_random = ['r','b','g']
    series_non_greedy = ['Wins(Non Greedy Play)','Loses(Non Greedy Play)','Draws(Non Greedy Play)']
    colors_non_greedy = ['c','m','k']
    markers_random = ['+','.','o']
    f = open('results.csv','wb')
    f_random = open('result_random.csv','wb')
    writer = csv.writer(f)
    writer_random = csv.writer(f_random)
    writer_non_greedy = csv.writer(f)
    writer.writerow(series)
    writer_random.writerow(series_random)
    writer_non_greedy.writerow(series_non_greedy)
    perf = [[] for _ in range(len(series_self_play) +1)]
    perf_random = [[] for _ in range(len(series_random) +1)]
    perf_non_greedy = [[] for _ in range(len(series_self_play) +1)]
    
    print "Start Self Play Learning"
    for i in range(10000):
        if i % 10 == 0:
            print "Play Game against random opponent"
            probs = measure_performance_vs_random_player(p1,p2)
            writer.writerow(probs)
            f.flush()
            perf_non_greedy[0].append(i)
            for idx,x in enumerate(probs):
                perf_non_greedy[idx+1].append(x)
        winner = play(p1,p2)
        p1.episode_over(winner)
        p2.episode_over(winner)
    f.close()
    print "Over Self Play Learning"
    
    agent1 = Agent(1,lossval = 0)
    #agent1.epsilon = 0
    r1 = Agent(2,lossval = 0)
    r1.epsilon = 1
    print "Start Random Learning"
    for i in range(10000):
        if i % 10 == 0:
            print "Play Game"
            probs = measure_performance_vs_random_player(agent1,r1)
            writer_random.writerow(probs)
            f_random.flush()
            perf_random[0].append(i)
            for idx,x in enumerate(probs):
                perf_random[idx+1].append(x)
        winner = play(agent1,r1)
        agent1.episode_over(winner)
        r1.episode_over(winner)
    f_random.close()
   
    print "Over Random Learning"
    
    for i in range(1,len(perf_non_greedy)):
        plt.plot(perf_non_greedy[0],perf_non_greedy[i],label=series_self_play[i-1],color=colors_self_play[i-1])

    for i in range(1,len(perf_random)):
        plt.plot(perf_random[0],perf_random[i],label=series_random[i-1],color=colors_random[i-1])
    
    """
    agent2 = Agent(1,lossval = 0)
    #agent2.epsilon = 0
    r2 = Agent(2,lossval = 0)
    r2.epsilon = 1
    print "Start Random Learning"
    for i in range(10000):
        if i % 10 == 0:
            print "Play Game"
            probs = measure_performance_vs_random_player(agent2,r2)
            writer_non_greedy.writerow(probs)
            #f_random.flush()
            perf_non_greedy[0].append(i)
            for idx,x in enumerate(probs):
                perf_non_greedy[idx+1].append(x)
        winner = play(agent2,r2)
        agent2.episode_over(winner)
        r2.episode_over(winner)
    f_random.close()

    for i in range(1,len(perf_non_greedy)):
        plt.plot(perf_non_greedy[0],perf_non_greedy[i],label=series_non_greedy[i-1],color=colors_non_greedy[i-1])

    plt.title('Greedy vs Non Greedy Play')
    plt.xlabel('Episodes')
    plt.ylabel('Probability')
    plt.legend(loc='best')
    plt.savefig('result_epsilon_zero.png')
    """
    plt.title('Trained using Self Play vs Trained against Random Opponent')
    plt.xlabel('Episodes')
    plt.ylabel('Probability')
    plt.legend(loc='best')
    plt.savefig('result_selfplay_random.png')
