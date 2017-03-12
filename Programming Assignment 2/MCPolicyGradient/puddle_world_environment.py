import random
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
import time

class puddle_world_environment(Environment):
	WORLD_FREE = 0
	WORLD_OBSTACLE = 1
	WORLD_SMALL_PENALITY = 2
	WORLD_MEDIUM_PENALITY = 3
	WORLD_LARGE_PENALITY = 4
	WORLD_GOAL = 5
	WESTERLY_WIND = True
	randGenerator = random.Random()
	fixedStartState = False
	startRows = [6,7,11,12] # Rows of the possible start states
	startCols = [1,1,1,1] # Columns of the possible start states
	startRow = 6
	startCol = 1
	print_state_flag = False
	num_steps = 0
	Return = 0
	gamma = 0.9

	def env_init(self):
		self.map = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1],
					[1,0,0,0,0,0,0,0,0,0,0,0,0,1],
					[1,0,0,0,0,0,0,0,0,0,0,0,0,1],
					[1,0,0,0,2,2,2,2,2,2,0,0,0,1],
					[1,0,0,0,2,3,3,3,3,2,0,0,0,1],
					[1,0,0,0,2,3,4,4,3,2,0,0,0,1],
					[1,0,0,0,2,3,4,3,3,2,0,0,0,1],
					[1,0,0,0,2,3,4,3,2,2,0,0,0,1],
					[1,0,0,0,2,3,3,3,2,0,0,0,0,1],
					[1,0,0,0,2,2,2,2,2,0,0,0,0,1],
					[1,0,0,0,0,0,0,0,0,0,0,0,0,1],
					[1,0,0,0,0,0,0,0,0,0,0,0,0,1],
					[1,0,0,0,0,0,0,0,0,0,0,0,0,1],
					[1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
		return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 0.9 OBSERVATIONS INTS (0 195) ACTIONS INTS (0 3) REWARDS (-3.0 10.0)"

	def env_start(self):
		rnd = self.randGenerator.randint(0,len(self.startRows)-1)
		self.startRow = self.startRows[rnd];
		self.startCol = self.startCols[rnd];
		startValid = self.setAgentState(self.startRow,self.startCol)
		returnObs = Observation()
		returnObs.intArray = [self.calculateFlatState()]
		self.count_to_goal = 0
		self.Return = 0
		return returnObs

	def env_step(self,thisAction):
		assert len(thisAction.intArray) == 1,"Expected 1 integer action"
		assert thisAction.intArray[0] >= 0, "Expected action to be in [0,3]"
		assert thisAction.intArray[0] < 4, "Expected action to be in [0,3]"

		self.updatePosition(thisAction.intArray[0])

		theObs = Observation()
		theObs.intArray = [self.calculateFlatState()]

		returnRO = Reward_observation_terminal()
		returnRO.r = self.calculateReward()
		returnRO.o = theObs
		returnRO.terminal = self.checkCurrentTerminal()
		self.Return += pow(self.gamma,self.count_to_goal) * self.calculateReward()
		self.count_to_goal += 1
		return returnRO

	def env_cleanup(self):
		pass

	def env_message(self, inMessage):
		if inMessage.startswith("wind-off"):
			self.WESTERLY_WIND = False
			return "Message understood. Turning Westerly Wind Off"

		if inMessage.startswith("wind-on"):
			self.WESTERLY_WIND = True
			return "Message understood. Turning Westerly Wind On"

		if inMessage.startswith("set-goal-A"):
			self.map[1][12] = 5
			self.map[3][10] = 0 # Making other goal states as free
			self.map[7][8] = 2 # Making other goal states as free
			return "Message understood. Goal set to position A"

		if inMessage.startswith("set-goal-B"):
			self.map[1][12] = 0 # Making other goal states as free
			self.map[3][10] = 5 # Making other goal states as free
			self.map[7][8] = 2
			return "Message understood. Goal set to position B"

		if inMessage.startswith("set-goal-C"):
			self.map[1][12] = 0 # Making other goal states as free
			self.map[3][10] = 0 # Making other goal states as free
			self.map[7][8] = 5
			self.WESTERLY_WIND = False
			return "Message understood. Goal set to position C"

		if inMessage.startswith("print-state"):
			self.printState()
			return "Message understood. Printed the state"

		if inMessage.startswith("print-update-state"):
			self.print_state_flag = True
			return "Message understood. Printing state in update"

		if inMessage.startswith("Return"):
			return str(self.Return)

		if inMessage.startswith("num_of_steps"):
			return str(self.count_to_goal)


	def setAgentState(self, row, col):
		self.agentRow = row
		self.agentCol = col
		return self.checkValid(row,col) and not self.checkTerminal(row,col)

	def checkValid(self,row,col):
		valid = False
		numRows = len(self.map)
		numCols = len(self.map[0])
		if(row < numRows and row >= 0 and col < numCols and col >= 0):
			if self.map[row][col] != self.WORLD_OBSTACLE:
				valid = True
		return valid

	def checkTerminal(self,row,col):
		if(self.map[row][col] == self.WORLD_GOAL):
			#print "goal reached",self.count_to_goal
			return True
		return False

	def checkCurrentTerminal(self):
		return self.checkTerminal(self.agentRow,self.agentCol)

	def calculateFlatState(self):
		numCols = len(self.map)
		return self.agentRow * numCols + self.agentCol

	def updatePosition(self, theAction):
		newRow = self.agentRow
		newCol = self.agentCol
		rnd = self.randGenerator.random()
		if(theAction == 1): # Move left
			if(rnd > 0.1):
				newCol = self.agentCol - 1
			elif(rnd < 0.1 / 3.0):
				newCol = self.agentCol + 1
			elif(rnd > 0.1 / 3.0 and rnd < 0.2 / 3.0):
				newRow = self.agentRow + 1
			elif(rnd > 0.2 / 3.0 and rnd < 0.1):
				newRow = self.agentRow - 1

		if(theAction == 0): # Move Right
			if(rnd > 0.1):
				newCol = self.agentCol + 1
			elif(rnd < 0.1 / 3.0):
				newCol = self.agentCol - 1
			elif(rnd > 0.1 / 3.0 and rnd < 0.2 / 3.0):
				newRow = self.agentRow + 1
			elif(rnd > 0.2 / 3.0 and rnd < 0.1):
				newRow = self.agentRow - 1

		if(theAction == 2): # Move Down
			if(rnd > 0.1):
				newRow = self.agentRow + 1
			elif(rnd < 0.1 / 3.0): # Move Up
				newCol = self.agentCol + 1
			elif(rnd > 0.1 / 3.0 and rnd < 0.2 / 3.0): # Move Down
				newCol = self.agentCol - 1
			elif(rnd > 0.2 / 3.0 and rnd < 0.1):
				newRow = self.agentRow - 1

		if(theAction == 3): # Move Up
			if(rnd > 0.1):
				newRow = self.agentRow - 1
			elif(rnd < 0.1 / 3.0): # Move Up
				newCol = self.agentCol + 1
			elif(rnd > 0.1 / 3.0 and rnd < 0.2 / 3.0): # Move Down
				newCol = self.agentCol - 1
			elif(rnd > 0.2 / 3.0 and rnd < 0.1): # Move left
				newRow = self.agentRow + 1

		#print newRow - self.agentRow, newCol - self.agentCol

		'''
		if(self.checkValid(newRow,newCol)):
			self.agentRow = newRow;
			self.agentCol = newCol;
		'''

		if(self.WESTERLY_WIND and self.randGenerator.random() > 0.5):
			newCol = newCol + 1;

		if(self.checkValid(newRow,newCol)):
			self.agentRow = newRow;
			self.agentCol = newCol;

		if self.print_state_flag:
			self.printState()
			time.sleep(0.2)


	def calculateReward(self):
		if(self.map[self.agentRow][self.agentCol] == self.WORLD_GOAL):
			return 10.0
		if(self.map[self.agentRow][self.agentCol] == self.WORLD_SMALL_PENALITY):
			return -1.0
		if(self.map[self.agentRow][self.agentCol] == self.WORLD_MEDIUM_PENALITY):
			return -2.0
		if(self.map[self.agentRow][self.agentCol] == self.WORLD_LARGE_PENALITY):
			return -3.0
		return 0.0

	def printState(self):
		numRows = len(self.map)
		numCols = len(self.map[0])
		print "Agent is at: "+str(self.agentRow)+","+str(self.agentCol)

		for row in range(0,numRows):
			print
			for col in range(0,numCols):
				if self.agentRow==row and self.agentCol==col:
					print "A",
				else:
					if self.map[row][col] == self.WORLD_GOAL:
						print "G",
					if self.map[row][col] == self.WORLD_SMALL_PENALITY:
						print "#",
					if self.map[row][col] == self.WORLD_MEDIUM_PENALITY:
						print "$",
					if self.map[row][col] == self.WORLD_LARGE_PENALITY:
						print "@",
					if self.map[row][col] == self.WORLD_OBSTACLE:
						print "*",
					if self.map[row][col] == self.WORLD_FREE:
						print " ",
		print

if __name__=="__main__":
	EnvironmentLoader.loadEnvironment(puddle_world_environment())
