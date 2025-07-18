import wandb
import argparse
import numpy as np
import pandas as pd

from src.env import ShopEnv
from src.config import Config
from src.recommenders import (
    RLRecommender, 
    RandomRecommender, 
    PopularityRecommender
)
from src.utils import (
    load_catalog,
    username_to_user
)
from src.utils.logger import get_logger
from src.utils.callbacks import WandbCallback, NaNCallback, LogProbCallback
from stable_baselines3.common.callbacks import CallbackList

logger = get_logger(__name__, level="DEBUG")

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
        default="models/rl_recommender.zip",
        help="Path to save the trained model (default: src/models/rl_recommender.zip)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="simshop-rl",
        help="Weights & Biases project name (default: simshop-rl)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom run name for Weights & Biases"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    return parser.parse_args()

def train():
    """
    Train the RL recommender model.
    """
    args = parse_args()
    
    logger.info("Starting training process...")
    logger.info(f"Total timesteps: {args.total_timesteps}")
    logger.info(f"Model will be saved to: {args.save_model_path}")
    
    config = Config()
    catalog = load_catalog(config.get("catalog_path"), config.get("catalog_size"))

    env_params = {
        "catalog": catalog,
        "username_to_user": username_to_user,
        "users_subset": [
            "cheap_seeker",
            # "brand_lover",
            # "random_chooser",
            # "value_optimizer",
            # "familiarity_seeker",
            # "freshness_looker"
        ]
    }

    rl_recommender = RLRecommender()
    # rl_recommender.load_model("models/checkpoint_200k")
    
    # Initialize Weights & Biases
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "total_timesteps": args.total_timesteps,
                "save_model_path": args.save_model_path,
                "num_recommendations": config.get("num_recommendations"),
                "num_candidates": config.get("num_candidates"),
                "catalog_size": config.get("catalog_size"),
                "users_subset": env_params["users_subset"],
            },
            tags=["rl", "recommender", "training"]
        )
    
    combined_callback = CallbackList([WandbCallback(verbose=0), NaNCallback(verbose=0), LogProbCallback(1)])
    
    rl_recommender.train(
        env_params=env_params,
        num_recommendations=config.get("num_recommendations"),
        total_timesteps=args.total_timesteps,
        callback=combined_callback #WandbCallback(verbose=0) if not args.no_wandb else None
    )
    
    # Save model first
    rl_recommender.save_model(args.save_model_path)
    
    # Log final evaluation metrics
    if not args.no_wandb:
        # Log model architecture after training is complete
        total_params = sum(p.numel() for p in rl_recommender.model.policy.parameters())
        feature_extractor = rl_recommender.model.policy.features_extractor
        feature_extractor_params = sum(p.numel() for p in feature_extractor.parameters())

        wandb.config.update({"model_params": total_params})
        wandb.config.update({"extractor_params":feature_extractor_params})
        wandb.config.update({"model_architecture": rl_recommender.model.policy})
        
        # Save model as artifact
        artifact = wandb.Artifact("rl_model", type="model")
        artifact.add_file(args.save_model_path)
        wandb.log_artifact(artifact)
        
        # Comprehensive evaluation with bar charts
        logger.info("\nRunning comprehensive evaluation for wandb visualization...")
        results_df = comprehensive_evaluation(
            catalog=catalog.sample(config.get("catalog_size"), random_state=42),
            rl_model_path=args.save_model_path,
            num_episodes=100,
            num_recommendations=config.get("num_recommendations")
        )
        
        # Save evaluation results as artifact
        results_path = f"src/metrics/evaluation_results_{wandb.run.id}.csv"
        results_df.to_csv(results_path, index=False)
        eval_artifact = wandb.Artifact("evaluation_results", type="dataset")
        eval_artifact.add_file(results_path)
        wandb.log_artifact(eval_artifact)
        
        wandb.finish()
    
    rl_recommender.save_model(args.save_model_path)

def evaluate_recommender(recommender, env, num_recommendations, num_episodes=100):
    """
    Evaluate a single recommender on a specific environment.
    """
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
    return np.mean(total_rewards)


def comprehensive_evaluation(catalog, rl_model_path, num_episodes=100, num_recommendations=5):
    """
    Run comprehensive evaluation comparing all recommenders across all users.
    Returns a DataFrame with results and logs bar charts to wandb.
    """
    
    # Initialize recommenders
    name_to_recommender = {
        "Random": RandomRecommender(),
        "Popularity": PopularityRecommender(),
        "RL": RLRecommender()
    }
    name_to_recommender['RL'].load_model(rl_model_path)
    
    # Evaluate each recommender on each user type
    results = []
    for username, user in username_to_user.items():
        env = ShopEnv(catalog, user)
        for recommender_name, recommender in name_to_recommender.items():
            avg_reward = evaluate_recommender(recommender, env, num_recommendations, num_episodes)
            results.append({
                'Recommender': recommender_name,
                'User': username,
                'Average Reward': avg_reward
            })
            # print(f"  {recommender_name} on {username}: {avg_reward:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Create pivot table for easier visualization
    pivot_df = results_df.pivot(index='User', columns='Recommender', values='Average Reward')
    
    # Create bar charts for each user type using wandb.plot.bar
    for user in pivot_df.index:
        # Prepare data for this user's bar chart
        bar_data = []
        for recommender in pivot_df.columns:
            reward = pivot_df.loc[user, recommender]
            bar_data.append([recommender, reward])
        
        # Create bar chart for this user
        table = wandb.Table(data=bar_data, columns=["Recommender", "Average Reward"])
        wandb.log({
            f"evaluation/bar_chart_{user}": wandb.plot.bar(
                table, 
                "Recommender", 
                "Average Reward",
                title=f"Recommender Performance for {user.replace('_', ' ').title()}"
            )
        })
    
    # Create overall summary bar chart with averaged results across users
    summary_data = []
    for recommender in pivot_df.columns:
        mean_reward = pivot_df[recommender].mean()
        summary_data.append([recommender, mean_reward])
    
    summary_table = wandb.Table(data=summary_data, columns=["Recommender", "Average Reward"])
    wandb.log({
        "evaluation/bar_chart_summary": wandb.plot.bar(
            summary_table,
            "Recommender",
            "Average Reward", 
            title="Overall Recommender Performance (Averaged Across All Users)"
        )
    })
    
    
    # Print summary table
    # print("\nSummary Results:")
    # print(pivot_df)
    # print(f"\nMean across users:")
    # for recommender in pivot_df.columns:
    #     mean_val = pivot_df[recommender].mean()
    #     std_val = pivot_df[recommender].std()
    #     print(f"  {recommender}: {mean_val:.3f} ± {std_val:.3f}")
    
    # print("Comprehensive evaluation completed!")
    return results_df

if __name__ == "__main__":
    train()


