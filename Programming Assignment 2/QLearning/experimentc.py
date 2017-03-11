import sys
import math
import numpy as np
import rlglue.RLGlue as RLGlue

num_of_agents = 50
num_of_episode = 500
goal_states = ['C']

for i in range(len(goal_states)):
	for j in range(num_of_agents):
		RLGlue.RL_init()
		RLGlue.RL_env_message("set-goal-" + goal_states[i])
		RLGlue.RL_env_message("wind-off")
		RLGlue.RL_agent_message("set-epsilon 0.55")
		returns = []
		num_steps = []
		for episode in range(num_of_episode):
			RLGlue.RL_episode(0)
			if(episode % 50 == 0):
				RLGlue.RL_agent_message("decay-epsilon")
			print 'Episode No: ' + str(episode + 1) + " Agent No: " + str(j + 1) + 'Goal :' + goal_states[i] + " Completed"
			returns.append(RLGlue.RL_return())
			num_steps.append(RLGlue.RL_num_steps())
		returns = np.array(returns)
		num_steps = np.array(num_steps)		
		np.save('returns/return_trial_' + 'agent_' + str(j) + '_goal_' + goal_states[i],returns)
		np.save('num_steps/num_step_trial_' + 'agent_' + str(j) + '_goal_' + goal_states[i],num_steps)

