import sys
import math
import rlglue.RLGlue as RLGlue
import numpy as np

sarsa_lambda = [0.0 , 0.3 , 0.5 , 0.9 , 0.99 , 1.0]
num_agent = 50
num_episodes = 500
goal_states = ['A','B']

for i in range(len(goal_states)):
	for j in range(len(sarsa_lambda)):
		for k in range(num_agent):
			returns = list()
			num_steps = list()
			RLGlue.RL_init()
			RLGlue.RL_env_message("set-goal-" + str(goal_states[i]))
			RLGlue.RL_agent_message("set_lambda " + str(sarsa_lambda[j]))
			for episode in range(num_episodes):
				RLGlue.RL_episode(0)
				print "Episode : " + str(episode + 1) + " Agent : " + str(k + 1) + " Lambda : " + str(sarsa_lambda[j]) + "Goal :" + goal_states[i] + " Completed"
				returns.append(RLGlue.RL_return())
				num_steps.append(RLGlue.RL_num_steps())
			returns = np.array(returns)
			num_steps = np.array(num_steps)
			np.save('returns/return_agent_' + str(k) + '_sarsa_lambda_' + str(sarsa_lambda[j]) + '_goal_state_' + goal_states[i] ,returns)
			np.save('num_steps/num_steps_agent_' + str(k) + '_sarsa_lambda_' + str(sarsa_lambda[j]) + '_goal_state_' + goal_states[i] ,num_steps)
