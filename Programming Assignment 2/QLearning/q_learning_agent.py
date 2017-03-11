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

class q_learning_agent(Agent):
	randGenerator = Random()
	lastAction = Action()
	lastObservation = Observation()
	q_learning_step_size = 0.1
	q_learning_epsilon = 0.1
	q_learning_gamma = 0.9
	numStates = 0
	numActions = 0
	value_function = None

	policyFrozen = False
	exploringFrozen = False

	def agent_init(self, taskSpecString):
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
			self.q_learning_gamma = TaskSpec.getDiscountFactor()
			self.value_function=[self.numActions*[0.0] for i in range(self.numStates)]

		else:
			print "Task Spec could not be parsed: "+ taskSpecString;
			
		self.lastAction=Action()
		self.lastObservation=Observation()

	def egreedy(self,state):
		if not self.exploringFrozen and random.uniform(0,1) < max(self.q_learning_epsilon,0.1):
			return self.randGenerator.randint(0,self.numActions - 1)
		return self.value_function[state].index(max(self.value_function[state]))

	def agent_start(self, observation):
		theState = observation.intArray[0]
		thisIntAction = self.egreedy(theState)
		returnAction = Action()
		returnAction.intArray = [thisIntAction]

		self.lastAction = copy.deepcopy(returnAction)
		self.lastObservation = copy.deepcopy(observation)

		return returnAction

	def agent_step(self,reward,observation):
		newState = observation.intArray[0]
		lastState = self.lastObservation.intArray[0]
		lastAction = self.lastAction.intArray[0]

		Q_sa = self.value_function[lastState][lastAction]
	
		new_Q_sa = Q_sa + self.q_learning_step_size * (reward + self.q_learning_gamma * max(self.value_function[newState]) - Q_sa)

		if not self.policyFrozen:
			self.value_function[lastState][lastAction] = new_Q_sa

		newIntAction = self.egreedy(newState)

		returnAction = Action()
		returnAction.intArray = [newIntAction]

		self.lastAction = copy.deepcopy(returnAction)
		self.lastObservation = copy.deepcopy(observation)

		return returnAction

	def agent_end(self,reward):
		lastState = self.lastObservation.intArray[0]
		lastAction = self.lastAction.intArray[0]

		Q_sa = self.value_function[lastState][lastAction]

		new_Q_sa = Q_sa + self.q_learning_step_size * (reward - Q_sa)

		if not self.policyFrozen:
			self.value_function[lastState][lastAction] = new_Q_sa

	def agent_cleanup(self):
		pass

	def save_value_function(self,fileName):
		theFile = open(fileName,"w")
		pickle.dump(self.value_function,theFile)
		theFile.close()

	def load_value_function(self,fileName):
		theFile = open(fileName,"r")
		self.value_function = pickle.load(theFile)
		theFile.close()

	def agent_message(self,inMessage):
		if inMessage.startswith("freeze learning"):
			self.policyFrozen=True
			return "message understood, policy frozen"

		if inMessage.startswith("unfreeze learning"):
			self.policyFrozen=False
			return "message understood, policy unfrozen"

		if inMessage.startswith("freeze exploring"):
			self.exploringFrozen=True
			return "message understood, exploring frozen"

		if inMessage.startswith("unfreeze exploring"):
			self.exploringFrozen=False
			return "message understood, exploring frozen"

		if inMessage.startswith("reduce exploring"):
			self.q_learning_epsilon = 0.3

		if inMessage.startswith("increase exploring"):
			self.q_learning_epsilon = 1.0

		if inMessage.startswith("decay-epsilon"):
			self.q_learning_epsilon -= 1/20.0
			print self.q_learning_epsilon

		if inMessage.startswith("set-epsilon"):
			splitString=inMessage.split(" ");
			self.q_learning_epsilon = float(splitString[1]);

		if inMessage.startswith("save_policy"):
			splitString=inMessage.split(" ");
			self.save_value_function(splitString[1]);
			print "Value Function Saved."
			return "message understood, saving policy"

		if inMessage.startswith("load_policy"):
			splitString=inMessage.split(" ")
			self.load_value_function(splitString[1])
			print "Loaded."
			return "message understood, loading policy"

		return "Q-Learning Agent(Python) does not understand your message."

if __name__ == "__main__":
	AgentLoader.loadAgent(q_learning_agent())
