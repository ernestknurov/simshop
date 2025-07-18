import numpy as np
import pandas as pd

from src.users import *
from src.config import Config

config = Config()

def load_catalog(path: str, size: int) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['release_date'])
    df['days_since_release'] = (pd.Timestamp.now() - df['release_date']).dt.days
    return df.sample(size)

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
