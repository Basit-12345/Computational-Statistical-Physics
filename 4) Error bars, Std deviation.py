import numpy as np
import matplotlib.pyplot as plt

num_walks = 400  # Number of walks in each simulation
max_steps = 70  # Maximum number of steps
d = 5             # Step length
start = 0         # Starting point
num_simulations = 20  # Number of times to repeat the simulation for error bars

num_steps_list = np.arange(0, max_steps + 1, 1)  # Array of the number of steps
mean_std_dev = np.zeros(len(num_steps_list))  # Store mean standard deviation for each simulation
std_dev_error = np.zeros(len(num_steps_list))  # Store error bars (std dev of std devs)
Expt = np.zeros(len(num_steps_list))  # Store expectation values

# Run the simulation multiple times and collect standard deviations
for num_steps in num_steps_list:
    if num_steps == 0:
        mean_std_dev[num_steps] = 0  # At 0 steps, standard deviation is 0
        std_dev_error[num_steps] = 0  # Error is also 0 at 0 steps
        continue

    std_devs = []  # List to store the standard deviations of each simulation for this number of steps
    
    # Run multiple simulations
    for _ in range(num_simulations):
        # Generate random steps (+d or -d) for each walk
        random_steps = np.random.choice([-d, d], size=(num_walks, num_steps))
        
        # Compute the cumulative sum of random steps
        
        
        # Add `start` as the first position by concatenating it as the first column
        random_steps = np.hstack((np.full((num_walks, 1), start), start + random_steps))
        
        
        walks = np.cumsum(random_steps, axis=1)
        
        # Take the last position of each walk
        end_points = walks[:, -1]
        
        # Calculate the standard deviation of the final positions for this simulation
        std_devs.append(np.std(end_points))
    
    # Calculate the mean standard deviation and its error (std of std deviations)
    mean_std_dev[num_steps] = np.mean(std_devs)
    std_dev_error[num_steps] = np.std(std_devs)

    # Calculate the expectation value of the final position
    Expt[num_steps] = np.mean(end_points)

# Calculate theoretical standard deviation for comparison
theoretical_std_dev = d * np.sqrt(num_steps_list)

# Plot the average std deviation with error bars representing the std deviation of std deviations
plt.figure(figsize=(10, 6))
plt.plot(num_steps_list, theoretical_std_dev, linestyle='--', color="red", label='Theoretical value')
plt.errorbar(num_steps_list, mean_std_dev, yerr=std_dev_error, fmt='o-', markersize=6, label='Observations from numerical simulations', 
             ecolor='black', capsize=6, capthick=1, elinewidth=2)  # Error bars more visible
plt.plot(num_steps_list, Expt, label='Expectation value of the final position')
plt.title("Standard Deviation of End Points vs. Number of Steps (with Error Bars)")
plt.xlabel("Number of Steps")
plt.ylabel("Standard Deviation of End Points")
plt.legend()
plt.grid(True)
plt.show()
