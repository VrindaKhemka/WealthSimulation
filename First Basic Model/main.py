import matplotlib.pyplot as plt
import agentModule as am
import simulation as s

# Set the parameters for the simulation
num_agents = 100
initial_wealth = 1000
num_time_steps = 1000
probability_dist = (0.05, 0.95)

sim = s.Simulation(num_agents, initial_wealth, num_time_steps, probability_dist, 0.1)
sim.run_simulation()

# Get the final wealth distribution and print some statistics
wealth_distribution = [agent.money for agent in sim.agents]
print(f"Minimum wealth: {min(wealth_distribution)}, Maximum wealth: {max(wealth_distribution)}")
print(f"Mean wealth: {sum(wealth_distribution) / num_agents:.2f}, Median wealth: {sorted(wealth_distribution)[num_agents//2]:.2f}")

# Plot a histogram of the wealth distribution
plt.hist(wealth_distribution, bins=50)
plt.title("Wealth Distribution")
plt.xlabel("Wealth")
plt.ylabel("Count")
plt.show()