import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium.spaces import MultiBinary, Dict, Box, Discrete

from typing import Tuple
from src.users import User 

class ShopEnv(gym.Env):
    """GYM environment for recommender agent interacting with users."""
    def __init__(self, user: User, items: pd.DataFrame) -> None:
        """Initialize environment with a user model and item catalog.
        Args:
            user (User): User behavior model.
            items (pd.DataFrame): Catalog of items to recommend.
        """
        self.user = user
        self.items = items
        self.encoded_items = self.encode_items(items)
        self.candidates = self._get_candidates()
        self.num_users = 5 # Hardcoding the the number of unique users for now
        self.item_feature_dim = 30 # hardcoding for now
        self.items_per_page = 10 

        self.click_weight = 0.5
        self.buy_weight = 1.0

        # Action is boolean vector with 1s on the ids of chosen items from catalog
        self.action_space = MultiBinary(len(items))
        self.observation_space = Dict({
            "user": Discrete(self.num_users), # For know encoding only user, later it past interactions too
            "history": Dict({
                "page_count": Box(low=0, high=np.inf, shape=(), dtype=np.int32), # Number of pages visited
                "click_count": Box(low=0, high=np.inf, shape=(), dtype=np.int32), # Number of clicks
                "buy_count": Box(low=0, high=np.inf, shape=(), dtype=np.int32), # Number of buys
                "last_click_item": Discrete(len(items)), # Last clicked item id
                "consecutive_no_click_pages": Box(low=0, high=np.inf, shape=(), dtype=np.int32), # Consecutive pages without clicks
            }),
            "candidates": Box(low=-np.inf, high=+np.inf, shape=(len(self.encoded_items), self.item_feature_dim)), # Matrix N items x F_dim features
        }) 
        self.done_criteria = {
            "consecutive_no_click_pages": 3,  # End if no clicks for 3 consecutive pages
            "page_count": 10,                  # End after 10 pages
            "click_count": 10,                 # End after 10 clicks
            "buy_count": 5                      # End after 5 buys
        }
        self.shown_items = set()  # Track shown items to avoid duplicates

    def reset(self) -> dict:
        """Reset state and return initial observation."""
        self.user.reset()

        self.history = {
            "page_count": 0,
            "click_count": 0,
            "buy_count": 0,
            "last_click_item": None,
            "consecutive_no_click_pages": 0,
        }
        self.shown_items.clear()
        self.candidates = self._get_candidates()

        return self.get_observation()
    
    def _get_candidates(self) -> pd.DataFrame:
        """Retrieve current candidate items."""
        return self.encoded_items
    
    def encode_items(self, items: pd.DataFrame) -> pd.DataFrame:
        """Convert raw items into feature-encoded representation."""
        return items
    
    def get_observation(self) -> dict:
        """Compile and return observation dict for agent."""
        return {
            "user": self.user.username,  # Assuming user has a username attribute
            "history": self.history,
            "candidates": self.candidates
        }


    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """Execute action, update state, compute reward, and return (obs, reward, done, info)."""
        done = False
        action_indices = np.where(action)[0]
        items_to_show = self.items.loc[action_indices]
        # REWARD CALCULATION
        clicked_items, bought_items = self.user.react(items_to_show)
        ctr, btr = clicked_items.mean(), bought_items.mean()
        reward = self.click_weight * ctr + self.buy_weight * btr
        info = {
            "clicked_items": clicked_items,
            "bought_items": bought_items,
            "recommended_items": items_to_show,
            "click_through_rate": ctr,
            "buy_through_rate": btr,
        }

        # STATE UPDATE
        self.shown_items.update(np.where(action)[0])  # Update shown items with the current action
        self.candidates = self._get_candidates()
        self.history["page_count"] += 1

        if ctr > 0:
            self.history["click_count"] += 1
            self.history["last_click_item"] = np.where(clicked_items)[0][-1] if clicked_items.any() else None
            self.history["consecutive_no_click_pages"] = 0
        else:
            self.history["consecutive_no_click_pages"] += 1
        if btr > 0:
            self.history["buy_count"] += 1
        
        # DONE CONDITIONS
        done_conditions = [
            self.history["consecutive_no_click_pages"] >= self.done_criteria["consecutive_no_click_pages"],
            self.history["page_count"] >= self.done_criteria["page_count"],
            self.history["click_count"] >= self.done_criteria["click_count"],
            self.history["buy_count"] >= self.done_criteria["buy_count"]
        ]
        done = any(done_conditions)

        return self.get_observation(), reward, done, info

