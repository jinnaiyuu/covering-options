import random
from collections import defaultdict

class QLearning():
    def __init__(self, gamma=0.95, learning_rate=0.1, epsilon=0.2):
        self.Qtable = defaultdict(lambda: defaultdict(lambda: -10000))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def act(self, state, actions):
        assert(len(actions) > 0)

        # ACTING MODULE
        return self.select_action(state, actions)


    def learn(self, prev_state, action, reward, state):
        if not self.Qtable[state]:
            s_qval = 0
        else:
            s_qval = max(self.Qtable[state])
        self.Qtable[prev_state][action] = (reward + self.gamma * s_qval) * self.learning_rate + self.Qtable[prev_state][action] * (1.0 - self.learning_rate)

    def select_action(self, state, actions):
        if random.random() > self.epsilon:
            # Greedy
            bestA = None
            bestValue = float('-inf')
            for a in actions:
                val = self.Qtable[state][a]
                if val > bestValue:
                    bestA = a
                    bestValue = val
            return bestA
        else:
            return random.choice(actions)

class RandAgent():
    def __init__(self):
        pass

    def act(self, state, actions):
        assert(len(actions) > 0)
        return random.choice(actions)
    

    def learn(self, prev_state, action, reward, state):
        pass
