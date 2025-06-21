import numpy as np
import pandas as pd

from src.users import *
from src.config import Config

config = Config()

def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['release_date'])
    df['release_days'] = (df['release_date'] - pd.Timestamp("1970-01-01")).dt.days
    return df

def action_to_indices(action):
    """
    Convert the action vector to indices of selected items.
    """
    return np.where(action == 1)[0].tolist()

def snake_case_to_camel_case(snake_str: str) -> str:
    return''.join(part.capitalize() for part in snake_str.split('_'))

username_to_user = {
    user: globals()[snake_case_to_camel_case(user) + 'User'](user, **params)
    for user, params in config.get("user_params").items()
}
