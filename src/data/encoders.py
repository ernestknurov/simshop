import numpy as np
import pandas as pd
from typing import List

def encode_items(items: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Encode categorical features in the item DataFrame."""
    df = items.copy()
    df['release_days'] = (df['release_date'].max() - df['release_date']).dt.days
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=np.int8)
    df.drop(columns=['release_date', 'name', 'description'], inplace=True)
    
    return df

def user_to_one_hot(username: str, users_list: List[str]) -> List[int]:
    """Convert a username to a one-hot encoded vector based on the users list."""
    one_hot = np.zeros(len(users_list), dtype=np.int8)
    if username in users_list:
        index = users_list.index(username)
        one_hot[index] = 1
    return one_hot