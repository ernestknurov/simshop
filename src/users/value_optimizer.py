import numpy as np
import pandas as pd
from .base import User

    
class ValueOptimizerUser(User):
    """
    A simulated user that seeks the optimal price-to-quality ratio (User Type D from project docs).
    
    This user type represents rational consumers who evaluate items based on value -
    the relationship between quality and price. They prefer items that offer the
    best quality per unit of price, making them quality-conscious but price-aware.
    
    Behavior characteristics:
        - Calculates utility as a scaled quality_score / price ratio
        - Seeks maximum value (quality per dollar spent)
        - Ignores brand, color, and other aesthetic preferences
        - Uses a scaling factor to ensure utility scores are in the 0-1 range
        - Represents rational, value-conscious consumers in the simulation
    
    Note:
        This user type requires items to have both 'quality_score' and 'price' columns.
        The quality_score should be a numeric measure of item quality.
    
    Example:
        >>> user = ValueOptimizerUser("value_seeker", click_threshold=0.4, buy_threshold=0.8)
        >>> # User will prefer high-quality items at reasonable prices
        >>> # Will click on items with above-threshold value ratios
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
        Calculate utility based on a scaled quality-to-price ratio (value optimization).
        
        Computes the value ratio for each item and scales it using a factor to ensure
        utility scores are in the 0-1 range. Items with higher quality and lower prices
        receive higher utility scores.
        
        Args:
            items: DataFrame containing item features (must include 'quality_score' and 'price' columns)
            
        Returns:
            Series of utility scores (0-1 range) based on scaled quality/price ratios
            
        Formula:
            scaled_value = (quality_score / price) * FACTOR
            utility = scaled_value / (1 + scaled_value)
        """
        FACTOR = 50
        # FACTOR is used to scale the score to get in the range of 0-1
        score = items['quality_score'] / items['price'] * FACTOR
        normalized_score = score / (1 + score)
        return normalized_score
