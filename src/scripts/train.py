import argparse

from src.config import Config
from src.recommenders import RLRecommender
from src.utils import (
    load_catalog,
    username_to_user
)


def parse_args():
    """
    Parse command line arguments for training.
    """
    parser = argparse.ArgumentParser(description="Train the RL recommender model")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2048,
        help="Total timesteps for training (default: 2048)"
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        default="src/models/rl_recommender.zip",
        help="Path to save the trained model (default: src/models/rl_recommender.zip)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for step-by-step environment interaction"
    )
    return parser.parse_args()

def train():
    """
    Train the RL recommender model.
    """
    args = parse_args()
    
    print("Starting training process...")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Model will be saved to: {args.save_model_path}")
    
    config = Config()
    catalog = load_catalog(config.get("catalog_path"))#.sample(50, random_state=42)

    env_params = {
        "catalog": catalog,
        "username_to_user": username_to_user,
        "users_subset": [
            "cheap_seeker",
            # "brand_lover",
            # "random_chooser",
            # "value_optimizer",
            # "familiarity_seeker"
        ]
    }

    rl_recommender = RLRecommender()
    rl_recommender.train(
        env_params=env_params,
        num_recommendations=config.get("num_recommendations"),
        total_timesteps=args.total_timesteps,
        debug=args.debug
    )
    rl_recommender.save_model(args.save_model_path)

if __name__ == "__main__":
    train()


