import random
import sys
import copy
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
import numpy as np

class sarsa_agent(Agent):
	randGenerator=Random()
	lastAction=Action()
	lastObservation=Observation()
	stepsize = 0.1
	gamma = 1.0
	numStates = 0
	numActions = 0
	value_function = None
	
	numRows = 12
	numCols = 12
	theta = None
	del_theta = None

	return_val = 0
	num_steps = 0

	temperature = 1.0

	policyFrozen=False
	exploringFrozen=False
	
	def agent_init(self,taskSpecString):
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
		if TaskSpec.valid:
			assert len(TaskSpec.getIntObservations())==1, "expecting 1-dimensional discrete observations"
			assert len(TaskSpec.getDoubleObservations())==0, "expecting no continuous observations"
			assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][0]), " expecting min observation to be a number not a special value"
			assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][1]), " expecting max observation to be a number not a special value"
			self.numStates=TaskSpec.getIntObservations()[0][1]+1;

			assert len(TaskSpec.getIntActions())==1, "expecting 1-dimensional discrete actions"
			assert len(TaskSpec.getDoubleActions())==0, "expecting no continuous actions"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
			assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
			self.numActions=TaskSpec.getIntActions()[0][1]+1;

			self.sarsa_gamma = TaskSpec.getDiscountFactor();
			
			#self.value_function=[self.numActions*[0.0] for i in range(self.numStates)]
			self.theta = np.zeros((self.numRows + self.numCols, self.numActions))
			self.del_theta = np.zeros((self.numRows + self.numCols, self.numActions))

		else:
			print "Task Spec could not be parsed: "+taskSpecString;
			
		self.lastAction=Action()
		self.lastObservation=Observation()
		

	def egreedy(self, state):
		if not self.exploringFrozen and self.randGenerator.random()<self.sarsa_epsilon:
			return self.randGenerator.randint(0,self.numActions-1)                
		#return self.value_function[state].index(max(self.value_function[state]))

	def getRowCol(self,state):
		col = state % (self.numCols + 2) 
		row = int(state / (self.numCols + 2))
		return (row - 1,col - 1)
	
	def softmaxActionSelection(self,state):
		(row, col) = self.getRowCol(state)
		action_prob = np.exp((self.theta[row] + self.theta[col + self.numRows])/ self.temperature) 
		prob = action_prob / np.sum(action_prob)
		return np.random.choice(range(4), p = prob)

	def softmax(self,theta):
		return np.exp(theta)/np.sum(np.exp(theta))
		
	
	def agent_start(self,observation):
		self.del_theta[:,:] = 0
		self.num_steps = 1
		self.return_val = 0
		theState = observation.intArray[0]
		#thisIntAction=self.egreedy(theState)
		thisIntAction = self.softmaxActionSelection(theState)
		returnAction = Action()
		returnAction.intArray = [thisIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)

		return returnAction
	
	def agent_step(self,reward, observation):
		newState=observation.intArray[0]
		lastState=self.lastObservation.intArray[0]
		lastAction=self.lastAction.intArray[0]

		newIntAction = self.softmaxActionSelection(newState)
		
		if not self.policyFrozen:
			self.return_val += reward * pow(self.gamma,self.num_steps - 1) 
			self.num_steps += 1
			(row,col) = self.getRowCol(lastState)
			self.del_theta[row] += (-1/self.temperature) * self.softmax(self.del_theta[row]/self.temperature)
			self.del_theta[col + self.numRows] += (-1/self.temperature) * self.softmax(self.del_theta[col + self.numRows]/self.temperature)
			self.del_theta[row, lastAction] += 1/self.temperature
			self.del_theta[col + self.numRows,lastAction] += 1/self.temperature 

		returnAction=Action()
		returnAction.intArray=[newIntAction]
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)

		return returnAction
	
	def agent_end(self,reward):
		lastState=self.lastObservation.intArray[0]
		lastAction=self.lastAction.intArray[0]

		if not self.policyFrozen:
			#self.value_function[lastState][lastAction]=new_Q_sa
			self.return_val += reward * pow(self.gamma,self.num_steps - 1) 
			self.num_steps += 1
			(row,col) = self.getRowCol(lastState)
			self.del_theta[row] += (-1/self.temperature) * self.softmax(self.del_theta[row]/self.temperature)
			self.del_theta[col + self.numRows] += (-1/self.temperature) * self.softmax(self.del_theta[col + self.numRows]/self.temperature)
			self.del_theta[row, lastAction] += 1/self.temperature
			self.del_theta[col + self.numRows,lastAction] += 1/self.temperature 

			self.theta += self.stepsize * self.return_val * self.del_theta

	
	def agent_cleanup(self):
		pass

	'''
	def save_value_function(self, fileName):
		theFile = open(fileName, "w")
		pickle.dump(self.value_function, theFile)
		theFile.close()

	def load_value_function(self, fileName):
		theFile = open(fileName, "r")
		self.value_function=pickle.load(theFile)
		theFile.close()
	'''

	def save_theta(self, fileName):
		theFile = open(fileName, "w")
		pickle.dump(self.theta, theFile)
		theFile.close()

	def load_theta(self, fileName):
		theFile = open(fileName, "r")
		self.theta = pickle.load(theFile)
		theFile.close()
	

	def agent_message(self,inMessage):
		
		if inMessage.startswith("freeze learning"):
			self.policyFrozen=True
			return "message understood, policy frozen"

		
		if inMessage.startswith("unfreeze learning"):
			self.policyFrozen=False
			return "message understood, policy unfrozen"

		'''
		if inMessage.startswith("freeze exploring"):
			self.exploringFrozen=True
			return "message understood, exploring frozen"

		
		if inMessage.startswith("unfreeze exploring"):
			self.exploringFrozen=False
			return "message understood, exploring frozen"
		'''
		
		if inMessage.startswith("save_policy"):
			splitString=inMessage.split(" ");
			#self.save_value_function(splitString[1]);
			self.save_theta(splitString[1]);
			print "Saved.";
			return "message understood, saving policy"

		if inMessage.startswith("load_policy"):
			splitString=inMessage.split(" ")
			#self.load_value_function(splitString[1])
			self.load_theta(splitString[1])
			print "Loaded."
			return "message understood, loading policy"

		return "SampleSarsaAgent(Python) does not understand your message."



if __name__=="__main__":
	AgentLoader.loadAgent(sarsa_agent())
