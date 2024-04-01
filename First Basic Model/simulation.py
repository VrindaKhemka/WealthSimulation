import random
import agentModule as am

class Simulation:

    def __init__(self, num_agents, initial_wealth, num_time_steps, probability_dist, buy_prob):
        self.num_agents = num_agents
        self.initial_wealth = initial_wealth
        self.num_time_steps = num_time_steps
        self.probability_dist = probability_dist
        self.buy_prob = buy_prob
        self.agents = [am.Agent(initial_wealth).allot_goods() for i in range(num_agents)]
    
    def update_agents(self):
        return
    
    def run_simulation(self):
        for i in range(self.num_time_steps):
            self.update_agents()

    def print_stats(self):
        total_wealth = sum(agent.money for agent in self.agents)
        avg_wealth = total_wealth / self.num_agents
        print(f"Total wealth: {total_wealth}, Average wealth: {avg_wealth}")