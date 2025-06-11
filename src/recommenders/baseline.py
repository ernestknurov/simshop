import numpy as np

class RandomRecommender:
    def recommend(self, state: dict, num_recommendations: int=10) -> np.ndarray:
        candidates = state['candidates']
        action = np.zeros(len(candidates), dtype=int)
        indices = np.random.choice(len(candidates), size=num_recommendations, replace=False)
        action[indices] = 1
        return action
    
class PopularityRecommender:
    def recommend(self, state: dict, num_recommendations: int=10) -> np.ndarray:
        candidates = state['candidates'].copy()
        candidates = candidates.sort_values(by='popularity', ascending=False)
        indices = candidates.index[:num_recommendations]
        action = np.zeros(len(candidates), dtype=int)
        action[indices] = 1
        return action