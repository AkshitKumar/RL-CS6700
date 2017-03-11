import sys
import math
import numpy as np
import rlglue.RLGlue as RLGlue

num_of_agents = 5
num_of_episode = 1000
goal_states = ['A']

for i in range(len(goal_states)):
	for j in range(num_of_agents):
		RLGlue.RL_init()
		RLGlue.RL_env_message("set-goal-" + goal_states[i])
		returns = []
		num_steps = []
		count_to_goal = []
		for episode in range(num_of_episode):
			RLGlue.RL_episode(0)
			returns.append(float(RLGlue.RL_env_message("Return")))
			num_steps.append(RLGlue.RL_num_steps())
			count_to_goal.append(int(RLGlue.RL_env_message("num_of_steps")))
			print 'Episode No: ' + str(episode + 1) + " Agent No: " + str(j + 1) + ' Goal :' + goal_states[i] + " Completed" + " Return : " + RLGlue.RL_env_message("Return") + "  Steps to Goal : " + RLGlue.RL_env_message("num_of_steps")
			if episode == num_of_episode - 1:
				RLGlue.RL_agent_message("save_policy result_" + str(j+1) + ".dat") # Save the value function for the agent
		returns = np.array(returns)
		num_steps = np.array(num_steps)		
		count_to_goal = np.array(count_to_goal)
		np.save('returns_goal_a/return_trial_' + 'agent_' + str(j) + '_goal_' + goal_states[i],returns)
		np.save('num_steps_goal_a/num_step_trial_' + 'agent_' + str(j) + '_goal_' + goal_states[i],num_steps)
		np.save('count_to_goal_a/num_step_trial_' + 'agent_' + str(j) + '_goal_' + goal_states[i],num_steps)
