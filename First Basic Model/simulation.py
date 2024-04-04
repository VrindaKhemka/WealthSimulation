import random
import agentModule as am

class Simulation:

    def __init__(self, num_agents, initial_wealth, num_time_steps, probability_dist, buy_prob):
        self.num_agents = num_agents
        self.initial_wealth = initial_wealth
        self.num_time_steps = num_time_steps
        self.probability_dist = probability_dist
        self.buy_prob = buy_prob
        self.agents = [am.Agent(initial_wealth) for _ in range(num_agents)]

    def update_agents(self):
        random_indices = random.sample(range(self.num_agents), 2)
        agent1, agent2 = self.agents[random_indices[0]], self.agents[random_indices[1]]

        if random.random() < self.buy_prob:
            if agent1.money >= agent2.price_of_product:
                agent1.buy_goods(agent2, 1, agent2.price_of_product)
                agent1.money -= agent2.price_of_product
                agent2.money += agent2.price_of_product

                # Randomly adjust the number of goods and price
                agent2.no_of_product += random.randint(-1, 1)
                agent2.price_of_product += random.uniform(-10, 10)

    def run_simulation(self):
        for _ in range(self.num_time_steps):
            self.update_agents()

    def print_stats(self):
        total_wealth = sum(agent.money for agent in self.agents)
        avg_wealth = total_wealth / self.num_agents
        print(f"Total wealth: {total_wealth}, Average wealth: {avg_wealth}")