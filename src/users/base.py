import numpy as np
import pandas as pd
from typing import Tuple

class User:
    """
    Base class for simulated user agents in the SimShop recommendation system.
    
    This class implements the core user behavior simulation framework where users
    react to recommended items through a two-stage decision process:
    1. Click decision: Based on whether item utility exceeds click_threshold
    2. Purchase decision: Probabilistic based on utility between click and buy thresholds
    
    The utility function defines how much a user likes an item (0-1 scale) and should
    be overridden by subclasses to implement specific user behaviors.
    
    Attributes:
        username (str): Unique identifier for the user
        click_threshold (float): Minimum utility score required for user to click an item
        buy_threshold (float): Utility score at which user has 100% probability to buy
    
    Note:
        - When utility >= click_threshold: User clicks the item
        - When click_threshold <= utility < buy_threshold: User buys with probability 
          proportional to how close utility is to buy_threshold
        - When utility >= buy_threshold: User always buys (100% probability)
    """
    
    def __init__(self, username: str, click_threshold: float, buy_threshold: float) -> None:
        """
        Initialize a user with specified behavior thresholds.
        
        Args:
            username: Unique identifier for the user
            click_threshold: Minimum utility score for clicking (0-1 range)
            buy_threshold: Utility score for guaranteed purchase (0-1 range)
            
        Note:
            buy_threshold should be >= click_threshold for realistic behavior
        """
        self.username = username
        self.click_threshold = click_threshold
        self.buy_threshold = buy_threshold

    def react(self, items: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate user reactions to a set of recommended items.
        
        This method implements the core user decision process:
        1. Calculate utility scores for all items
        2. Determine clicks based on click_threshold
        3. Determine purchases probabilistically based on buy_threshold
        
        Args:
            items: DataFrame containing item features (must include columns needed by utility function)
            
        Returns:
            Tuple of (clicked_items, bought_items) where each is a boolean array
            indicating which items were clicked/bought
            
        Note:
            - Items can only be bought if they were first clicked
            - Purchase probability is linear between click_threshold and buy_threshold
        """

        scores = self.utility(items)  # returns np.ndarray or Series

        clicked = (scores >= self.click_threshold).astype(np.uint8)

        # Avoid unnecessary pandas ops — use vectorized NumPy
        span = self.buy_threshold - self.click_threshold
        buy_prob = np.clip((scores - self.click_threshold) / span, 0, 1)
        bought = ((np.random.rand(len(scores)) < buy_prob) & (clicked == 1)).astype(np.uint8)

        return clicked, bought, scores.to_numpy()

    
    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Calculate utility scores for items (should be overridden by subclasses).
        
        Utility represents how much the user likes each item on a 0-1 scale:
        - 0: No interest in the item
        - 1: Maximum interest in the item
        
        Args:
            items: DataFrame containing item features
            
        Returns:
            Series of utility scores (0-1 range) for each item
            
        Note:
            This default implementation returns 0.5 for all items and should be
            overridden by subclasses to implement specific user preferences
        """
        return pd.Series([0.5] * len(items))  # Default utility function, should be overridden by subclasses
    
    def reset(self) -> None:
        """
        Reset the user state for a new session.
        
        This method should be called between sessions to clear any stateful
        information (e.g., seen items, interaction history) that shouldn't
        persist across different recommendation sessions.
        """
        pass
