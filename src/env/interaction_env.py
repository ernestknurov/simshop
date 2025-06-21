import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium.spaces import MultiBinary, Dict, Box, Discrete

from typing import Tuple
from src.users import User 
from src.data.encoders import encode_items, user_to_one_hot
from src.config import Config

config = Config()

class ShopEnv(gym.Env):
    """GYM environment for recommender agent interacting with users."""
    def __init__(self, items: pd.DataFrame, user: User) -> None:
        """Initialize environment with a user model and item catalog.
        Args:
            items (pd.DataFrame): Catalog of items to recommend.
        """
        super().__init__()
        self.items = items
        self.user = user  # Factory to create user instances

        self.shown_items = set()  # Track shown items to avoid duplicates
        self.cat_features = config.get("catalog")["cat_features"]
 
        self.num_features = config.get("catalog")["num_features"]
        self.encoded_items = encode_items(items, self.cat_features)
        self.one_hot_cat_features = [col for col in self.encoded_items.columns if col.startswith(tuple(self.cat_features))]
        self.one_hot_user = user_to_one_hot(self.user.username, config.get("users_list"))

        self.num_candidates = config.get('num_candidates')
        self.num_users = len(config.get("users_list"))

        self.items_per_page = 10 
        self.coverage = 0.0 # percentage of items shown to user
        self.ctr = 0.0  # Click-through rate
        self.btr = 0.0  # Buy-through rate
        self.episode_count = 0

        self.click_weight = 1.0
        self.buy_weight = 2.0

        # Action is boolean vector with 1s on the ids of chosen items from catalog
        self.action_space = MultiBinary(self.num_candidates)

        self.observation_space = Dict({
            "user": Box(low=0, high=1, shape=(len(self.one_hot_user),), dtype=np.int8),
            "candidates_cat_features": Box(low=0, high=1, shape=(self.num_candidates, len(self.one_hot_cat_features)), dtype=np.int8),
            "candidates_num_features": Box(low=0.0, high=np.inf, shape=(self.num_candidates, len(self.num_features)), dtype=np.float32),
            # "history": Dict({
            #     "page_count": Box(low=0, high=np.inf, shape=(), dtype=np.int32), # Number of pages visited
            #     "click_count": Box(low=0, high=np.inf, shape=(), dtype=np.int32), # Number of clicks
            #     "buy_count": Box(low=0, high=np.inf, shape=(), dtype=np.int32), # Number of buys
            #     "last_click_item": Discrete(len(items)), # Last clicked item id
            #     "consecutive_no_click_pages": Box(low=0, high=np.inf, shape=(), dtype=np.int32), # Consecutive pages without clicks
            # }),
        }) 
        self.done_criteria = {
            "consecutive_no_click_pages": 3,  
            "page_count": 10,                  
            "click_count": 10,                 
            "buy_count": 5                      
        }

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        """Reset state and return initial observation."""
        super().reset(seed=seed)
        self.history = {
            "page_count": 0,
            "click_count": 0,
            "buy_count": 0,
            "last_click_item": None,
            "consecutive_no_click_pages": 0,
        }
        if self.episode_count % 10 == 0: # HARDCODED
            self.user.reset()
        self.shown_items.clear()
        self.candidates = self._get_candidates()
        self.coverage = 0.0
        self.ctr = 0.0
        self.btr = 0.0
        self.episode_count += 1

        return self.get_observation(), {}
    
    def _get_candidates(self) -> pd.DataFrame:
        """Retrieve current candidate items."""
        # choose random candidates from available items
        available_items = self.encoded_items[~self.encoded_items.product_id.isin(self.shown_items)]
        candidates = available_items.sample(n=self.num_candidates, replace=False)
        return candidates.reset_index(drop=True)
    
    
    def get_observation(self) -> dict:
        """Compile and return observation dict for agent."""
        return {
            "user": self.one_hot_user, 
            "candidates_cat_features": self.candidates[self.one_hot_cat_features].values.astype(np.int8),
            "candidates_num_features": self.candidates[self.num_features].values.astype(np.float32),
            # "history": self.history,
        }


    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Execute action, update state, compute reward, and return (obs, reward, done, truncated, info)."""
        done = False

        # TAKE ACTION
        action_indices = np.where(action)[0]
        # print(f"Action vector length: {len(action)}, 1s count: {np.sum(action)}")
        items_to_show = self.candidates.loc[action_indices] # encoded items
        items_to_show = self.items.loc[self.items['product_id'].isin(items_to_show['product_id'])].reset_index(drop=True) # original items
        clicked_items, bought_items = self.user.react(items_to_show)

        # REWARD CALCULATION
        ctr, btr = clicked_items.mean(), bought_items.mean()
        reward = self.click_weight * ctr + self.buy_weight * btr
        info = {
            "recommended_items": items_to_show,
            "clicked_items": clicked_items,
            "bought_items": bought_items,
            "recommended_items": items_to_show,
            "click_through_rate": ctr,
            "buy_through_rate": btr,
            "history": self.history,
        }

        # STATE UPDATE
        self.history["page_count"] += 1
        self.shown_items.update(items_to_show.product_id.tolist())  # Update shown items with the current action
        self.coverage = len(self.shown_items) / len(self.encoded_items)
        self.ctr = self.ctr + (ctr - self.ctr) / self.history["page_count"]
        self.btr = self.btr + (btr - self.btr) / self.history["page_count"]
        self.candidates = self._get_candidates()

        if ctr > 0:
            self.history["click_count"] += sum(clicked_items)
            self.history["last_click_item"] = items_to_show.iloc[np.where(clicked_items)[0][-1]].product_id
            self.history["consecutive_no_click_pages"] = 0
        else:
            self.history["consecutive_no_click_pages"] += 1
        if btr > 0:
            self.history["buy_count"] += sum(bought_items)
        
        # DONE CONDITIONS
        done_conditions = [
            self.history["consecutive_no_click_pages"] >= self.done_criteria["consecutive_no_click_pages"],
            self.history["page_count"] >= self.done_criteria["page_count"],
            self.history["click_count"] >= self.done_criteria["click_count"],
            self.history["buy_count"] >= self.done_criteria["buy_count"]
        ]
        done = any(done_conditions)

        return self.get_observation(), reward, done, False, info

