import argparse
from datetime import datetime
import time

import numpy as np
import pandas as pd

from src.config import Config
from src.utils import (
    load_catalog,
    username_to_user
)
from src.utils.logger import get_logger
from src.env import ShopEnv
from src.recommenders import (
    RandomRecommender,
    PopularityRecommender,
    RLRecommender
)

logger = get_logger(__name__, level="DEBUG")

def parse_args():
    """
    Parse command line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate the recommenders")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Num episodes to run for evaluation (default: 100)"
    )
    parser.add_argument(
        "--rl-model-path",
        type=str,
        default="models/rl_recommender.zip",
        help="Path to load the trained rl model (default: models/rl_recommender.zip)"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=f"metrics/recommender_comparison_{datetime.now().strftime('%Y_%m_%d-%H_%M')}.csv",
        help="Path to save the evaluation metrics (default: metrics/recommender_comparison_{current_data}.csv)"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Whether show logs or not. 0 - no, 1 - yes (default: 0)."
    )
    return parser.parse_args()

def evaluate_recommender(recommender, env, num_recommendations, num_episodes=100):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = recommender.recommend(state, num_recommendations=num_recommendations)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    average_reward = np.mean(total_rewards)
    return average_reward


def compare_recommenders(catalog, name_to_recommender, num_episodes, num_recommendations, verbose):
    report = pd.DataFrame(columns=['Recommender', 'User', 'Average Reward'])
    for username, user in username_to_user.items():
        env = ShopEnv(catalog, user)
        for recommender_name, recommender in name_to_recommender.items():
            avg_reward = evaluate_recommender(recommender, env, num_recommendations, num_episodes)
            if verbose:
                print(f"Recommender: {recommender_name}, User: {username}, Average Reward: {avg_reward}")
            report.loc[len(report)] = {
                'Recommender': recommender_name,
                'User': username,
                'Average Reward': avg_reward
            }
    return report

def main():
    start_time = time.time()

    args = parse_args()
    logger.info("\nStarting evaluation process...")
    logger.info(f"Eval episodes: {args.eval_episodes}")
    logger.info(f"RL model will be loaded from: {args.rl_model_path}")
    logger.info(f"Metrics will be saved to: {args.results_path}")

    config = Config()
    catalog = load_catalog(config.get("catalog_path"), 250)#.sample(50, random_state=42)

    name_to_recommender = {
        "random": RandomRecommender(),
        "popularity": PopularityRecommender(),
        "rl": RLRecommender()
    }
    name_to_recommender['rl'].load_model(args.rl_model_path)
    report = compare_recommenders(catalog, name_to_recommender, args.eval_episodes, config.get("num_recommendations"), args.verbose)
    logger.info(report)
    report.to_csv(args.results_path, index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()