import pandas as pd
from .base import User

class FreshnessLookerUser(User):
    """
    A user type that prefers newer items based on their freshness (days since release).
    This user calculates utility inversely proportional to how old an item is relative
    to other items in the dataset. Newer items receive higher utility scores.
    Attributes:
        username (str): The username of the user
        click_threshold (float): Threshold for clicking on items
        buy_threshold (float): Threshold for purchasing items
    """
    
    def __init__(self, username: str, click_threshold: float, buy_threshold: float) -> None:
        """
        Initialize a FreshnessLookerUser.
        Args:
            username (str): The username of the user
            click_threshold (float): Threshold value for clicking behavior
            buy_threshold (float): Threshold value for buying behavior
        """
        super().__init__(username, click_threshold, buy_threshold)

    def utility(self, items: pd.DataFrame) -> pd.Series:
        """
        Calculate utility scores for items based on their freshness.
        Items that were released more recently receive higher utility scores.
        The utility is calculated as 1 minus the normalized oldness, where
        oldness is the relative position of an item's release date within
        the range of all items' release dates.
        Args:
            items (pd.DataFrame): DataFrame containing items with 'days_since_release' column
        Returns:
            pd.Series: Utility scores for each item, where higher values indicate
                        higher preference (newer items get higher scores)
        """
        span = max(items['days_since_release'].max() - items['days_since_release'].min(), 1)
        oldness = (items['days_since_release'] - items['days_since_release'].min()) / span
        return 1 - oldness
    