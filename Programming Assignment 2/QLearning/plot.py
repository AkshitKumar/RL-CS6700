import numpy as np
import matplotlib.pyplot as plt

goal_states = ['A','B','C']
num_agent = 50
return_array_A = np.zeros(1000)
return_array_B = np.zeros(1000)
return_array_C = np.zeros(500)
num_steps_A = np.zeros(1000)
num_steps_B = np.zeros(1000)
num_steps_C = np.zeros(500)

for i in range(num_agent):
	return_array_A += np.load('returns/return_agent_' + str(i) + '_goal_' + 'A' + '.npy')
	return_array_B += np.load('returns/return_agent_' + str(i) + '_goal_' + 'B' + '.npy')
	num_steps_A += np.load('num_steps/num_step_' + 'agent_' + str(i) + '_goal_' + 'A.npy')
	num_steps_B += np.load('num_steps/num_step_' + 'agent_' + str(i) + '_goal_' + 'B.npy')
	return_array_C += np.load('returns/return_trial_agent_' + str(i) + '_goal_' + 'C' + '.npy')
	num_steps_C += np.load('num_steps/num_step_trial_' + 'agent_' + str(i) + '_goal_' + 'C.npy')

return_array_A = return_array_A / float(num_agent)
return_array_B = return_array_B / float(num_agent)
return_array_C = return_array_C / float(num_agent)
num_steps_A = num_steps_A / float(num_agent)
num_steps_B = num_steps_B / float(num_agent)
num_steps_C = num_steps_C / float(num_agent)
x = np.arange(500)
x += 1
#plt.plot(x,return_array_A,x,return_array_B)
plt.plot(x,return_array_C)
print return_array_C	
#print num_steps_B
plt.show()
