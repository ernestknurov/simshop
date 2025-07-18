import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium.spaces import MultiBinary, Dict, Box, Discrete

from typing import Tuple
from src.users import User 
from src.data.encoders import encode_items_with_embeddings, user_to_one_hot
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
        self.user = user

        self.shown_items = set()
        self.cat_features = config.get("catalog")["cat_features"]
 
        self.num_features = config.get("catalog")["num_features"]
        items_decay_rate = config.get("user_params")["freshness_looker"]["decay_rate"]
        self.encoded_items, self.vocab_mappings = encode_items_with_embeddings(items, self.cat_features, items_decay_rate)
        self.one_hot_user = user_to_one_hot(self.user.username, config.get("users_list"))

        self.num_candidates = config.get('num_candidates')
        self.num_users = len(config.get("users_list"))
        self.n_last_clicks = config.get("n_last_clicks")

        self.items_per_page = config.get("num_recommendations") 
        self.coverage = 0.0 # percentage of items shown to user
        self.ctr = 0.0  # Click-through rate
        self.btr = 0.0  # Buy-through rate
        self.episode_count = 0

        self.click_weight = 3.0
        self.buy_weight = 6.0

        # Action is boolean vector with 1s on the ids of chosen items from catalog
        self.action_space = MultiBinary(self.num_candidates)

        self.observation_space = Dict({
            "user": Box(low=0, high=1, shape=(len(self.one_hot_user),), dtype=np.int8),
            "candidates_cat_features": Box(low=0, high=127, shape=(self.num_candidates, len(self.cat_features)), dtype=np.int8),
            "candidates_num_features": Box(low=0.0, high=np.inf, shape=(self.num_candidates, len(self.num_features)), dtype=np.float32),
            "history_n_last_click_items_cat_features": Box(low=0, high=127, shape=(self.n_last_clicks, len(self.cat_features)), dtype=np.int8),
            "history_n_last_click_items_num_features": Box(low=0.0, high=np.inf, shape=(self.n_last_clicks, len(self.num_features)), dtype=np.float32),
            "history_n_last_click_items_mask": Box(low=0, high=1, shape=(self.n_last_clicks,), dtype=np.int8)
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
        self.n_last_click_items = pd.DataFrame(
            0, 
            index=range(self.n_last_clicks), 
            columns=self.cat_features + self.num_features, 
            dtype=np.float32
        )
        self.n_last_click_items_mask = np.zeros(self.n_last_clicks, dtype=np.int8)

    def reset(self, seed=None, options=None) -> Tuple[dict, dict]:
        """Reset state and return initial observation."""
        super().reset(seed=seed)
        self.history = {
            "page_count": 0,
            "click_count": 0,
            "buy_count": 0,
            "last_click_item": None,
            "consecutive_no_click_pages": 0,
            # "num_items_to_show": -1
        }
        if self.episode_count % 10 == 0: # HARDCODED
            self.user.reset()
        self.shown_items.clear()
        self.candidates = self._get_candidates()
        self.coverage = 0.0
        self.ctr = 0.0
        self.btr = 0.0
        self.episode_count += 1
        self.done = False

        return self.get_observation(), {}
    
    def _get_candidates(self) -> pd.DataFrame:
        """Retrieve current candidate items."""
        # choose random candidates from available items
        available_items = self.encoded_items
        # available_items = self.encoded_items[~self.encoded_items.product_id.isin(self.shown_items)]
        candidates = available_items.sample(n=self.num_candidates, replace=False)
        return candidates.reset_index(drop=True)
    
    
    def get_observation(self) -> dict:
        """Compile and return observation dict for agent."""
        return {
            "user": self.one_hot_user, 
            "candidates_cat_features": self.candidates[self.cat_features].values.astype(np.int8),
            "candidates_num_features": self.candidates[self.num_features].values.astype(np.float32),
            "history_n_last_click_items_cat_features": self.n_last_click_items[self.cat_features].values.astype(np.int8),
            "history_n_last_click_items_num_features": self.n_last_click_items[self.num_features].values.astype(np.float32),
            "history_n_last_click_items_mask": self.n_last_click_items_mask,
        }

    # @profile
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Execute action, update state, compute reward, and return (obs, reward, done, truncated, info)."""
        if self.done:
            raise RuntimeError("Environment is done. Please reset it before calling step().")

        # TAKE ACTION
        action_indices = np.where(action)[0]
        items_to_show = self.candidates.loc[action_indices] # encoded items
        if not len(items_to_show):
            print("[WARNING] Items to show len is null")
        items_to_show = self.items.loc[self.items['product_id'].isin(items_to_show['product_id'])].reset_index(drop=True) # original items
        clicked_items, bought_items, utility_scores = self.user.react(items_to_show)

        # REWARD CALCULATION
        utility_score_weight = 0.1
        if len(utility_scores):
            utility_reward = utility_score_weight * max(utility_scores)
        else:
            utility_reward = 0
            print("[WARNING] Utitility scores are null")
        ctr, btr = clicked_items.mean(), bought_items.mean()
        reward = self.click_weight * ctr + self.buy_weight * btr + utility_reward
        if ctr == 0:
            reward -= 0.1  # small penalty for no clicks
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
        if ctr > 0:
            n_last_clicks_items_ids = items_to_show[clicked_items.astype(bool)].product_id
            candidates_mask = self.candidates.product_id.isin(n_last_clicks_items_ids)
            self.n_last_click_items = pd.concat([self.n_last_click_items, self.candidates[candidates_mask]]).tail(self.n_last_clicks)
            if any(1 - self.n_last_click_items_mask): # if there are empty slots in n_last_click_items_mask
                num_clicks = sum(clicked_items)
                self.n_last_click_items_mask = np.roll(self.n_last_click_items_mask, -num_clicks)
                self.n_last_click_items_mask[-num_clicks:] = 1
        self.candidates = self._get_candidates()

        if ctr > 0:
            self.history["click_count"] += sum(clicked_items)
            self.history["last_click_item"] = items_to_show.iloc[np.where(clicked_items)[0][-1]].product_id
            self.history["consecutive_no_click_pages"] = 0
        else:
            self.history["consecutive_no_click_pages"] += 1
        if btr > 0:
            self.history["buy_count"] += sum(bought_items)
        # self.history['num_items_to_show'] = len(items_to_show)
        
        # DONE CONDITIONS
        done_conditions = [
            self.history["consecutive_no_click_pages"] >= self.done_criteria["consecutive_no_click_pages"],
            self.history["page_count"] >= self.done_criteria["page_count"],
            self.history["click_count"] >= self.done_criteria["click_count"],
            self.history["buy_count"] >= self.done_criteria["buy_count"]
        ]
        self.done = any(done_conditions)

        return self.get_observation(), reward, self.done, False, info

