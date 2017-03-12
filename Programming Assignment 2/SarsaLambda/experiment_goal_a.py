import sys
import math
import numpy as np
import rlglue.RLGlue as RLGlue

num_of_agents = 50
num_of_episode = 1000
goal_states = ['A']
lambdas = [0.0, 0.3, 0.9 , 0.99 , 1.0]

def evaluateAgent(eval_return,eval_num_of_step):
	calc_return = 0
	calc_num_steps = 0	
	n = 10
	RLGlue.RL_agent_message("freeze learning")
	RLGlue.RL_agent_message("freeze exploring")
	for i in range(0,n):
		RLGlue.RL_episode(5000)
		calc_return += float(RLGlue.RL_env_message("Return"))
		calc_num_steps += int(RLGlue.RL_env_message("num_of_steps"))
		print "EVALUATION Episode : " + str(i) + " RETURN : " + RLGlue.RL_env_message("Return") + " STEPS : " + RLGlue.RL_env_message("num_of_steps")
	RLGlue.RL_agent_message("unfreeze exploring")
	RLGlue.RL_agent_message("unfreeze learning")
	calc_return /= n
	calc_num_steps /= n
	eval_return.append(calc_return)
	eval_num_of_step.append(calc_num_steps)

def saveEvaluation(eval_return,eval_num_of_step,num,lambda_val):
	eval_num_of_step = np.array(eval_num_of_step)
	eval_return = np.array(eval_return)
	np.save('eval_goal_a/eval_return_agent_' + str(num) + '_lambda_' + str(lambda_val),eval_return)
	np.save('eval_goal_a/eval_num_steps_agent_' + str(num) '_lambda_' + str(lambda_val),eval_num_of_step)

for i in range(len(goal_states)):
	for index,lambda_val in enumerate(lambdas):
		for j in range(num_of_agents):
			RLGlue.RL_init()
			RLGlue.RL_env_message("set-goal-" + goal_states[i])
			RLGlue.RL_agent_message("set_lambda " + str(lambda_val))
			returns = []
			num_steps = []
			eval_return = []
			eval_num_of_step = []
			for episode in range(num_of_episode):
				if (episode % 25 == 0):
					evaluateAgent(eval_return,eval_num_of_step)
				RLGlue.RL_episode(0)
				returns.append(float(RLGlue.RL_env_message("Return")))
				num_steps.append(RLGlue.RL_num_steps())
				print 'Episode No: ' + str(episode + 1) + " Agent No: " + str(j + 1) + ' Goal :' + goal_states[i] + " Lambda : "+ str(lambda_val) + " Completed" + " Return : " + RLGlue.RL_env_message("Return") + "  Steps to Goal : " + RLGlue.RL_env_message("num_of_steps")
				if episode == num_of_episode - 1:
					RLGlue.RL_agent_message("save_policy result_goal_A_agent_" + str(j+1) + ".dat") # Save the value function for the agent
			returns = np.array(returns)
			num_steps = np.array(num_steps)		
			np.save('returns_goal_a/return_' + 'agent_' + str(j) + '_lambda_' + str(lambda_val) + '_goal_' + goal_states[i],returns)
			np.save('num_steps_goal_a/num_step_' + 'agent_' + str(j) + '_lambda_' + str(lambda_val) + '_goal_' + goal_states[i],num_steps)
			saveEvaluation(eval_return,eval_num_of_step,j,lambda_val)