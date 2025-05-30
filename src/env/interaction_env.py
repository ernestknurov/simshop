import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiBinary, Dict, Box, Discrete

class Shop(gym.Env):
    """GYM like environment for recommender agent to interact with users."""
    def __init__(self, user, items):
        self.user = user
        # self.recommender = recommender
        self.items = items
        self.encoded_items = self.encode_items(items)
        # Action is boolean vector with 1s on the ids of chosen items from catalog
        self.action_space = MultiBinary(len(items))
        self.num_users = 5 # Hardcoding the the number of unique users for now
        self.item_feature_dim = len(self.encoded_items[0]) # hardcoding for now

        self.observation_space = Dict({
            "user": Discrete(self.num_users), # For know encoding only user, later it past interactions too
            "candidates": Box(low=-np.inf, high=+np.inf, shape=(len(self.encoded_items), self.item_feature_dim)), # Matrix N items x F_dim features
        }) 

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.user.reset()
        return self.get_observation()
    
    def encode_items(self, items):
        return items
    
    def get_observation(self):
        return {
            "user": self.user,
            "candidates": np.array([item['features'] for item in self.items], dtype=np.float32)
        }

    def step(self, action):
        items_to_show = self.items[action]
        chosen_items = self.user_react(items_to_show)
        reward = 1.0 * len(chosen_items)
        pass

