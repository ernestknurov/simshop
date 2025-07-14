import numpy as np
import pandas as pd
from .base import User

class FreshnessLookerUser(User):
    """
    A specialized user type that evaluates items based on their freshness,
    defined as the number of days since their release. This user assigns higher
    utility scores to newer items, with the utility decaying exponentially as
    items become older.

    Attributes:
        username (str): The username of the user.
        click_threshold (float): The threshold for clicking on items, representing
                                 the minimum utility score required for the user to
                                 consider clicking on an item.
        buy_threshold (float): The threshold for purchasing items, representing
                               the minimum utility score required for the user to
                               consider buying an item.
        decay_rate (float): The rate at which the utility decays as the item's age increases.
    """
    
    def __init__(self, username: str, click_threshold: float, buy_threshold: float, decay_rate: float) -> None:
        """
        Initialize a FreshnessLookerUser.

        Args:
            username (str): The username of the user.
            click_threshold (float): Threshold value for clicking behavior.
            buy_threshold (float): Threshold value for buying behavior.
            decay_rate (float): The rate at which the utility decays as the item's age increases.
        """
        super().__init__(username, click_threshold, buy_threshold)
        self.decay_rate = decay_rate
    
    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Calculate utility scores for items based on their freshness.

        Items that were released more recently receive higher utility scores.
        The utility is calculated using an exponential decay function applied to
        the 'days_since_release' column of the input DataFrame.

        Args:
            items (pd.DataFrame): A DataFrame containing item data, which must include
                                  a 'days_since_release' column indicating the number
                                  of days since each item's release.

        Returns:
            pd.Series: A Series of utility scores for each item, where higher values
                       indicate greater preference for newer items.
        """
        return np.exp(-self.decay_rate * items['days_since_release'])
    