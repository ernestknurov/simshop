import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def encode_items_with_one_hot(items: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Encode categorical features in the item DataFrame."""
    df = items.copy()
    df['release_days'] = (df['release_date'].max() - df['release_date']).dt.days
    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, dtype=np.int8)
    df.drop(columns=['release_date', 'name', 'description'], inplace=True)
    
    return df

def encode_items_with_embeddings(items: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Encode categorical features as indices for embeddings instead of one-hot.
    Returns the dataframe and vocabulary mappings.
    """
    df = items.copy()
    df['release_days'] = (df['release_date'].max() - df['release_date']).dt.days
    df.drop(columns=['release_date', 'name', 'description'], inplace=True)
    
    # Create vocabulary mappings for each categorical feature
    vocab_mappings = {}
    
    for col in cat_cols:
        unique_values = sorted(df[col].unique())
        vocab_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
        # Convert to indices
        df[col] = df[col].map(vocab_mappings[col])
    
    return df, vocab_mappings

def user_to_one_hot(username: str, users_list: List[str]) -> List[int]:
    """Convert a username to a one-hot encoded vector based on the users list."""
    one_hot = np.zeros(len(users_list), dtype=np.int8)
    if username in users_list:
        index = users_list.index(username)
        one_hot[index] = 1
    return one_hot