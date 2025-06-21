import numpy as np

class RandomRecommender:
    def recommend(self, state: dict, num_recommendations: int=10) -> np.ndarray:
        num_candidates = state['candidates_num_features'].shape[0]
        action = np.zeros(num_candidates, dtype=int)
        indices = np.random.choice(num_candidates, size=num_recommendations, replace=False)
        action[indices] = 1
        return action
    
class PopularityRecommender:
    def recommend(self, state: dict, num_recommendations: int=10) -> np.ndarray:
        popularity = state['candidates_num_features'][:, 2]
        recommendations_indices = np.argsort(popularity)[::-1][:num_recommendations]
        action = np.zeros(len(popularity), dtype=int)
        action[recommendations_indices] = 1
        return action