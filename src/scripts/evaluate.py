import numpy as np
import pandas as pd

from src.config import Config
from src.utils import (
    load_catalog,
    username_to_user
)
from src.env import ShopEnv
from src.recommenders import (
    RandomRecommender,
    PopularityRecommender,
    RLRecommender
)


def evaluate_recommender(recommender, env, num_episodes=100):
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = recommender.recommend(state, num_recommendations=10)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    average_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {average_reward}")
    return average_reward


def compare_recommenders(catalog, name_to_recommender):
    report = pd.DataFrame(columns=['Recommender', 'User', 'Average Reward'])
    for username, user in username_to_user.items():
        env = ShopEnv(catalog, user)
        for recommender_name, recommender in name_to_recommender.items():
            print(f"Evaluating {recommender_name} recommender for user: {username}")
            avg_reward = evaluate_recommender(recommender, env, num_episodes=100)
            report.loc[len(report)] = {
                'Recommender': recommender_name,
                'User': username,
                'Average Reward': avg_reward
            }
    return report

def main():
    config = Config()
    catalog = load_catalog(config.get("catalog_path"))

    name_to_recommender = {
        "random": RandomRecommender(),
        "popularity": PopularityRecommender(),
        "rl": RLRecommender()
    }
    name_to_recommender['rl'].load_model('src/models/rl_recommender.zip')

    print("Starting evaluation of recommenders...")
    report = compare_recommenders(catalog, name_to_recommender)
    print(report)
    report.to_csv('src/metrics/recommender_comparison.csv', index=False)

if __name__ == "__main__":
    main()