# Code for the evolution of standard deviation with the number of steps.
import numpy as np
import matplotlib.pyplot as plt

num_walks = 1000  # Number of walks in each simulation
max_steps = 200  # Maximum number of steps
d = 5              # Step length
start = 0       # Starting point

num_steps_list = np.arange(0, max_steps + 1, 1)  # Array of the number of steps
std_dev = np.zeros(len(num_steps_list))  # Store std deviation for each simulation
Expt=np.zeros(len(num_steps_list)) 

for num_steps in num_steps_list:
        if num_steps == 0:
            std_dev[num_steps] = 0  # At 0 steps, standard deviation is 0
            continue

        # Generate random steps (+d or -d) for each walk
        random_steps = np.random.choice([-d, d], size=(num_walks, num_steps))
        
        # Add `start` as the first position by concatenating it as the first column
        random_steps=np.hstack((np.full((num_walks, 1), start), start + random_steps))
        
        # Compute the cumulative sum of random steps
        walks = np.cumsum(random_steps, axis=1)
        
        # Take the last position of each walk
        end_points = walks[:, -1]
        std_dev[num_steps] = np.std(end_points)# Calculate std deviation of end points
        Expt[num_steps]=np.mean(end_points) 
        
# Calculate theoretical standard deviation for comparison
theoretical_std_dev = d * np.sqrt(num_steps_list)

# Plot the average std deviation with error bars representing the std deviation of std deviations
plt.figure(figsize=(10, 6))
plt.plot(num_steps_list,theoretical_std_dev,linestyle='--', color="red", label='Theoretical value')
plt.plot(num_steps_list,std_dev,'o-',markersize=3, label='Observations from numerical simulations')
plt.plot(num_steps_list,Expt, label='expectation value of the final position')
plt.title("Standard Deviation of End Points vs. Number of Steps")
plt.xlabel("Number of Steps")
plt.ylabel("Standard Deviation of End Points")
plt.legend()
plt.grid(True)
plt.show()