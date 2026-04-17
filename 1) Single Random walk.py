#Code to obtain a single random walk
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_steps = 200 # Number of steps in the walk
d = 5               # Step length
start = 100      # Starting position

random_steps = np.random.choice([-d, d], size=num_steps)

steps = np.insert(random_steps, 0, start)

walk=np.cumsum(steps)
        
plt.plot(np.arange(0,num_steps+1,1),walk,'o-')
plt.title("The position of a random walker at each step", fontsize=14)
plt.xlabel("Steps",fontsize=14)
plt.ylabel("Position",fontsize=14)
plt.grid('True')
plt.show()
