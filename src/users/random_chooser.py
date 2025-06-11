import numpy as np
import pandas as pd
from .base import User

class RandomChooserUser(User):
    """
    A simulated user that makes completely random choices (User Type E from project docs).
    
    This user type represents consumers who make purchasing decisions without any
    systematic preferences or rational evaluation. They ignore all item features
    (price, quality, brand, color) and select items purely at random, serving as
    a baseline for comparison with other user types.
    
    Behavior characteristics:
        - Completely ignores all item features and attributes
        - Makes random decisions independent of price, quality, brand, or color
        - Utility scores are uniformly distributed between 0 and 1
        - Serves as a control group for evaluating recommendation algorithms
        - Represents impulsive or indifferent consumers in the simulation
    
    Use Cases:
        - Baseline comparison for recommendation system performance
        - Stress testing recommendation algorithms with unpredictable behavior
        - Simulating noise in user behavior datasets
        - Control group for A/B testing scenarios
    
    Example:
        >>> user = RandomChooserUser("random_buyer", click_threshold=0.5, buy_threshold=0.8)
        >>> # User will randomly click on ~50% of items and buy ~20% of clicked items
        >>> # Behavior is completely independent of item characteristics
    """
    
    def __init__(self, username: str, click_threshold: float, buy_threshold: float) -> None:
        """
        Initialize a random-choosing user.
        
        Args:
            username: Unique identifier for the user
            click_threshold: Minimum utility score for clicking (0-1 range)
            buy_threshold: Utility score for guaranteed purchase (0-1 range)
            
        Note:
            The thresholds determine what fraction of randomly-scored items will be
            clicked/bought, but the selection is still random within those fractions.
        """
        super().__init__(username, click_threshold, buy_threshold)

    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Generate completely random utility scores for all items.
        
        Returns uniformly distributed random scores between 0 and 1 for each item,
        completely ignoring all item features and characteristics. This creates
        purely random user behavior that serves as a baseline for comparison.
        
        Args:
            items: DataFrame containing item features (all columns are ignored)
            
        Returns:
            Series of random utility scores uniformly distributed in [0, 1] range
            
        Note:
            Each call generates new random scores, so the same items may receive
            different utility scores on different exposures.
        """
        items = items.copy()  # Ensure we don't modify the original DataFrame
        items['score'] = np.random.rand(len(items))  # Generate random scores
        return items['score']