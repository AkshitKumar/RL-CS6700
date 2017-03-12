import numpy as np
import matplotlib.pyplot as plt

goal_states = ['A','B','C']
num_agent = 50
return_array_A = np.zeros(1000)
num_steps_A = np.zeros(1000)


for i in range(num_agent):
	return_array_A += np.load('returns_goal_b/return_agent_' + str(i) + '_goal_' + 'B' + '.npy')
	num_steps_A += np.load('num_steps_goal_b/num_step_' + 'agent_' + str(i) + '_goal_' + 'B.npy')
	

return_array_A = return_array_A / float(num_agent)
num_steps_A = num_steps_A / float(num_agent)

x = np.arange(1000)
x += 1
#plt.plot(x,return_array_A,x,return_array_B)
plt.plot(x,return_array_A)
print return_array_A
#print num_steps_B
plt.show()
