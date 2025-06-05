import numpy as np

class RandomRecommender:
    def recommend(self, state):
        candidates = state['candidates']
        action = np.zeros(len(candidates), dtype=int)
        indices = np.random.choice(candidates.index, size=10, replace=False)
        action[indices] = 1
        return action
    