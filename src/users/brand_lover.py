import numpy as np
import pandas as pd
from .base import User

class BrandLoverUser(User):
    """
    A simulated user that prefers specific brands and colors (User Type B from project docs).
    
    This user type has strong preferences for certain brands and colors, representing
    consumers who are loyal to particular brands or have aesthetic preferences.
    The utility function combines brand and color preferences with configurable weights,
    plus optional noise to add behavioral variability.
    
    Behavior characteristics:
        - Strongly prefers items from specific brands (70% weight by default)
        - Has secondary preferences for certain colors (30% weight by default)
        - Ignores price and quality considerations
        - Adds random noise to simulate human behavioral inconsistency
        - Represents brand-loyal consumers in the simulation
    
    Attributes:
        brand_weights (dict): Mapping of brand names to preference scores (0-1)
        color_weights (dict): Mapping of color names to preference scores (0-1)
        noise (float): Standard deviation of random noise added to utility scores
    
    Example:
        >>> brand_prefs = {"Nike": 0.9, "Adidas": 0.8, "Generic": 0.2}
        >>> color_prefs = {"Black": 0.9, "White": 0.7, "Pink": 0.3}
        >>> user = BrandLoverUser("brand_fan", 0.4, 0.8, brand_prefs, color_prefs)
        >>> # User will strongly prefer Nike/Adidas items in Black/White colors
    """
    
    def __init__(self, username: str, click_threshold: float, buy_threshold: float, 
                 brand_weights: dict[str, float], color_weights: dict[str, float], noise: float=0.1) -> None:
        """
        Initialize a brand-loving user with specific brand and color preferences.
        
        Args:
            username: Unique identifier for the user
            click_threshold: Minimum utility score for clicking (0-1 range)
            buy_threshold: Utility score for guaranteed purchase (0-1 range)
            brand_weights: Dictionary mapping brand names to preference scores (0-1)
            color_weights: Dictionary mapping color names to preference scores (0-1)
            noise: Random noise standard deviation to add behavioral variability (default: 0.1)
        
        Note:
            Items with brands/colors not in the weights dictionaries get default score of 0.1
        """
        super().__init__(username, click_threshold, buy_threshold)
        
        self.brand_weights = brand_weights
        self.color_weights = color_weights
        self.noise = noise

    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Calculate utility based on brand and color preferences with added noise.
        
        Combines brand preferences (70% weight) and color preferences (30% weight)
        to calculate base utility, then adds Gaussian noise for behavioral realism.
        Unknown brands/colors receive a default low score of 0.1.
        
        Args:
            items: DataFrame containing item features (must include 'brand' and 'color' columns)
            
        Returns:
            Series of utility scores (0-1 range) based on brand/color preferences
            
        Formula:
            utility = 0.7 * brand_score + 0.3 * color_score + noise
            where noise ~ Uniform(-noise_std, +noise_std)
        """
        items = items.copy()
        items['brand_score'] = items['brand'].apply(lambda x: self.brand_weights.get(x, 0.1))
        items['color_score'] = items['color'].apply(lambda x: self.color_weights.get(x, 0.1))
        items['score'] = 0.7 * items['brand_score'] + 0.3 * items['color_score']
        items['score'] = items['score'] + np.random.uniform(-self.noise, self.noise, size=len(items))
        return items['score'].clip(0, 1)