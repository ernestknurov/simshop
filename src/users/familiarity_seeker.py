import numpy as np
import pandas as pd
from .base import User
from collections import defaultdict

def sigmoid(x: float, offset: float) -> float:
    """
    Sigmoid function to normalize scores for familiarity-based utility.
    
    Args:
        x: Input value (number of times item was seen)
        offset: Offset parameter to control the sigmoid curve shape
        
    Returns:
        Normalized score between 0 and 1
    """
    return 1 / (1 + np.exp(-x + offset))

class FamiliaritySeekerUser(User):
    """
    A simulated user that prefers familiar items seen multiple times (User Type C from project docs).
    
    This user type represents consumers who are hesitant about new products and need
    repeated exposure before developing interest. They ignore new products on first
    exposure but may click on them after seeing them multiple times, simulating
    the psychological "mere exposure effect" where familiarity breeds preference.
    
    Behavior characteristics:
        - Initially ignores new products (low utility on first exposure)
        - Utility increases with repeated item exposure via sigmoid function
        - Maintains stateful memory of previously seen items across interactions
        - Adds small Gaussian noise for behavioral realism
        - Represents cautious, familiarity-seeking consumers in the simulation
    
    Attributes:
        seen_items (defaultdict): Counter tracking how many times each product_id was seen
        noise_std (float): Standard deviation of Gaussian noise added to utility scores
    
    Mathematical Model:
        utility = sigmoid(seen_count, offset=4) + noise
        where noise ~ N(0, noise_std²)
    
    Example:
        >>> user = FamiliaritySeekerUser("cautious_buyer", 0.3, 0.7, noise_std=0.05)
        >>> # User will initially ignore new items but warm up to them over time
        >>> # After 4+ exposures, items become much more appealing
    """
    
    def __init__(
        self,
        username: str,
        click_threshold: float,
        buy_threshold: float,
        noise_std: float = 0.05
    ) -> None:
        """
        Initialize a familiarity-seeking user.
        
        Args:
            username: Unique identifier for the user
            click_threshold: Minimum utility score for clicking (0-1 range)
            buy_threshold: Utility score for guaranteed purchase (0-1 range)
            noise_std: Standard deviation of Gaussian noise for behavioral variability (default: 0.05)
        """
        super().__init__(username, click_threshold, buy_threshold)
        self.seen_items = defaultdict(int)
        self.noise_std = noise_std

    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Calculate utility based on item familiarity (exposure count) with noise.
        
        For each item, increments the seen count and calculates utility using a sigmoid
        function of the exposure count. Items seen more frequently receive higher utility
        scores, with small Gaussian noise added for behavioral realism.
        
        Args:
            items: DataFrame containing item features (must include 'product_id' column)
            
        Returns:
            Series of utility scores (0-1 range) based on familiarity and noise
            
        Algorithm:
            1. Increment seen_count for each product_id
            2. Apply sigmoid transformation: sigmoid(seen_count, offset=4)
            3. Add Gaussian noise: N(0, noise_std²)
            4. Clip final scores to [0, 1] range
            
        Note:
            The offset=4 means items become appealing after ~4 exposures
        """
        # 1) Increment the seen_count for each product_id
        items = items.copy()
        for _, row in items.iterrows():
            self.seen_items[row["product_id"]] += 1

        # 2) Compute the deterministic "familiarity" via sigmoid
        counts = items["product_id"].map(self.seen_items)
        base_scores = counts.apply(lambda c: sigmoid(c, offset=4))

        # 3) Add small Gaussian noise, then clip into [0, 1]
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=len(items))
        noisy_scores = (base_scores + noise).clip(0.0, 1.0)

        return noisy_scores
    
    def reset(self) -> None:
        """
        Reset the user state for a new session.
        
        Clears the seen_items counter, effectively making all items "new" again.
        This should be called between different recommendation sessions to ensure
        users start fresh without memory of previous sessions.
        """
        self.seen_items.clear()
        super().reset()

