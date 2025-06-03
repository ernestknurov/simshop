import numpy as np
import pandas as pd
from .base import User

    
class ValueOptimizerUser(User):
    """
    A simulated user that seeks optimal price-to-quality ratio (User Type D from project docs).
    
    This user type represents rational consumers who evaluate items based on value -
    the relationship between quality and price. They prefer items that offer the
    best quality per unit of price, making them quality-conscious but price-aware.
    
    Behavior characteristics:
        - Calculates utility as quality_score / price ratio
        - Seeks maximum value (quality per dollar spent)
        - Ignores brand, color, and other aesthetic preferences
        - Uses min-max normalization to ensure utility scores are in 0-1 range
        - Represents rational, value-conscious consumers in the simulation
    
    Note:
        This user type requires items to have both 'quality_score' and 'price' columns.
        The quality_score should be a numeric measure of item quality.
    
    Example:
        >>> user = ValueOptimizerUser("value_seeker", click_threshold=0.4, buy_threshold=0.8)
        >>> # User will prefer high-quality items at reasonable prices
        >>> # Will click on items with above-average value ratios
    """
    
    def __init__(self, username: str, click_threshold: float, buy_threshold: float) -> None:
        """
        Initialize a value-optimizing user.
        
        Args:
            username: Unique identifier for the user
            click_threshold: Minimum utility score for clicking (0-1 range)
            buy_threshold: Utility score for guaranteed purchase (0-1 range)
        """
        super().__init__(username, click_threshold, buy_threshold)

    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Calculate utility based on quality-to-price ratio (value optimization).
        
        Computes the value ratio for each item and normalizes using min-max scaling
        to ensure utility scores are in the 0-1 range. Items with higher quality
        and lower prices receive higher utility scores.
        
        Args:
            items: DataFrame containing item features (must include 'quality_score' and 'price' columns)
            
        Returns:
            Series of utility scores (0-1 range) based on normalized quality/price ratios
            
        Formula:
            raw_value = quality_score / price
            utility = (raw_value - min_value) / (max_value - min_value)
        """
        items = items.copy()
        items['score'] = items['quality_score'] / items['price']
        return (items['score'] - items['score'].min()) / (items['score'].max() - items['score'].min())
