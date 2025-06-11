import numpy as np
import pandas as pd
from .base import User

class CheapSeekerUser(User):
    """
    A simulated user that prioritizes low-priced items (User Type A from project docs).
    
    This user type always prefers the cheapest available items, implementing a 
    price-sensitive behavior where utility is inversely proportional to price.
    The user calculates utility using a pivot-based normalization formula:
    utility = pivot_price / (pivot_price + item_price), giving highest scores 
    to the cheapest items relative to the pivot price.
    
    Behavior characteristics:
        - Always chooses the cheapest item among recommendations
        - Utility decreases as price increases, normalized by pivot price
        - Completely ignores other item features (brand, color, quality, etc.)
        - Represents budget-conscious consumers in the simulation
    
    Example:
        >>> user = CheapSeekerUser("budget_buyer", click_threshold=0.3, buy_threshold=0.7, pivot_price=100)
        >>> # User will click on items with utility >= 0.3 (relatively cheap items)
        >>> # User will buy items with utility >= 0.7 (very cheap items)
    """
    
    def __init__(self, username: str, click_threshold: float, buy_threshold: float, pivot_price: float) -> None:
        """
        Initialize a cheap-seeking user.
        
        Args:
            username: Unique identifier for the user
            click_threshold: Minimum utility score for clicking (0-1 range)
            buy_threshold: Utility score for guaranteed purchase (0-1 range)
            pivot_price: Reference price for normalizing utility scores
        """
        super().__init__(username, click_threshold, buy_threshold)
        self.pivot_price = pivot_price

    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Calculate utility based on item price (lower price = higher utility).
        
        Implements the core cheap-seeking behavior by scoring items inversely
        to their price, normalized by the pivot price. The utility score is
        calculated as:
        
        utility = pivot_price / (pivot_price + item_price)
        
        Args:
            items: DataFrame containing item features (must include 'price' column)
            
        Returns:
            Series of utility scores (0-1 range) where lower prices get higher scores
        """
        return self.pivot_price / (self.pivot_price + items['price'])  # Normalize to pivot price
