import numpy as np
import pandas as pd
from .base import User

class CheapSeekerUser(User):
    """
    A simulated user that prioritizes low-priced items (User Type A from project docs).
    
    This user type always prefers the cheapest available items, implementing a 
    price-sensitive behavior where utility is inversely proportional to price.
    The user calculates utility as 1 - (item_price / max_price), giving highest
    scores to the cheapest items.
    
    Behavior characteristics:
        - Always chooses the cheapest item among recommendations
        - Utility decreases linearly with price increase
        - Completely ignores other item features (brand, color, quality, etc.)
        - Represents budget-conscious consumers in the simulation
    
    Example:
        >>> user = CheapSeekerUser("budget_buyer", click_threshold=0.3, buy_threshold=0.7)
        >>> # User will click on items with utility >= 0.3 (relatively cheap items)
        >>> # User will buy items with utility >= 0.7 (very cheap items)
    """
    
    def __init__(self, username: str, click_threshold, buy_threshold) -> None:
        """
        Initialize a cheap-seeking user.
        
        Args:
            username: Unique identifier for the user
            click_threshold: Minimum utility score for clicking (0-1 range)
            buy_threshold: Utility score for guaranteed purchase (0-1 range)
        """
        super().__init__(username, click_threshold, buy_threshold)

    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Calculate utility based on item price (lower price = higher utility).
        
        Implements the core cheap-seeking behavior by scoring items inversely
        to their price. The cheapest item gets utility score of 1.0, while
        the most expensive item gets utility score of 0.0.
        
        Args:
            items: DataFrame containing item features (must include 'price' column)
            
        Returns:
            Series of utility scores (0-1 range) where lower prices get higher scores
            
        Formula:
            utility = 1 - (item_price / max_price_in_set)
        """
        items = items.copy()
        items['score'] = 1 - (items['price'] / items['price'].max())
        return items['score'].clip(0, 1)
