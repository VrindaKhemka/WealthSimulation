import random

class Agent:

    def __init__(self):
        self.no_of_product = 0
        self.price_of_product = 0
        self.money = 0
    
    def __init__(self, initial_wealth):
        self.no_of_product = 0
        self.price_of_product = 0
        self.money = initial_wealth

    def allot_goods(self):
        self.no_of_product = random.randint(1, 10)  # allocate a random number of goods between 1 and 10
        self.price_of_product = random.randint(10, 100)  # allocate a random price between 10.00 and 100.00
    
    def allot_money(self):
        self.money = random.randint(100,500)


    def buy_goods(self, other_agent, num_goods, price):
        if random.random() < 0.1: # 10% chance of buying goods
            self.no_of_product += num_goods
            self.money -= num_goods * price
            other_agent.no_of_product -= num_goods
            other_agent.money += num_goods * price