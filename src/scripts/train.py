import wandb
import argparse
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

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


class WandbCallback(BaseCallback):
    """
    Custom callback for logging metrics to Weights & Biases during training.
    """
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.step_rewards = []
        self.current_episode_rewards = []
        self.current_episode_steps = 0
        self.ctr = []
        self.btr = []
        self.page_count = []
        self.consecutive_no_click_pages = []
        
    def _on_step(self) -> bool:
        # Track step-level rewards for all environments
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        # Handle rewards (could be array for multiple envs)
        if len(rewards) > 0:
            if isinstance(rewards, (list, np.ndarray)):
                step_reward = rewards[0]  # Take first environment
            else:
                step_reward = float(rewards)
            
            self.step_rewards.append(step_reward)
            self.current_episode_rewards.append(step_reward)
            self.current_episode_steps += 1
            
            # Log step metrics every 100 steps to reduce noise
            if self.num_timesteps % 100 == 0:
                wandb.log({
                    "training/step_reward": step_reward,
                    "training/mean_step_reward_100": np.mean(self.step_rewards[-100:]) if len(self.step_rewards) >= 100 else np.mean(self.step_rewards)
                })
        
        # Check for episode completion
        episode_ended = False
        episode_reward = 0
        episode_length = 0
    

        # Log custom environment metrics from info
        if infos:
            for info in infos:
                if isinstance(info, dict):
                    env_metrics = {}
                    
                    # Log environment-specific metrics
                    if 'click_through_rate' in info:
                        self.ctr.append(float(info.get('click_through_rate')))
                        env_metrics['env/mean_ctr_100'] = np.mean(self.ctr[-100:])
                    if 'buy_through_rate' in info:
                        self.btr.append(float(info.get('buy_through_rate')))
                        env_metrics['env/mean_btr_100'] = np.mean(self.btr[-100:])
                    if 'history' in info and isinstance(info['history'], dict):
                        history = info['history']
                        self.page_count.append(int(history.get('page_count', 0)))
                        self.consecutive_no_click_pages.append(int(history.get('consecutive_no_click_pages', 0)))
                        env_metrics.update({
                            'env/mean_page_count_100': np.mean(self.page_count[-100:]),
                            'env/mean_consecutive_no_click_pages_100': np.mean(self.consecutive_no_click_pages[-100:])
                        })
                    
                    if env_metrics:
                        wandb.log(env_metrics)
                    
                    if 'episode' in info:
                        episode_info = info['episode']
                        episode_reward = float(episode_info['r'])
                        episode_length = int(episode_info['l'])
                        episode_ended = True
                    break  # Only log from first environment to avoid duplicates
        
        
        # Log episode completion
        if episode_ended:
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_count += 1
            
            # Reset current episode tracking
            self.current_episode_rewards = []
            self.current_episode_steps = 0
            
            # Calculate rolling averages
            recent_rewards = self.episode_rewards[-100:]
            mean_reward = np.mean(recent_rewards)
            
            recent_lengths = self.episode_lengths[-100:]
            mean_length = np.mean(recent_lengths)
            
            # Log episode metrics to wandb
            wandb.log({
                "episode/reward": episode_reward,
                "episode/length": episode_length,
                "episode/count": self.episode_count,
                "episode/mean_reward_100": mean_reward,
                "episode/mean_length_100": mean_length,
                "episode/max_reward": max(self.episode_rewards) if self.episode_rewards else 0,
                "episode/min_reward": min(self.episode_rewards) if self.episode_rewards else 0,
            })
            
            if self.verbose > 0:
                print(f"Episode {self.episode_count}: Reward={episode_reward:.3f}, Length={episode_length}, Mean(100)={mean_reward:.3f}")
            
        return True


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
            "brand_lover",
            # "random_chooser",
            "value_optimizer",
            # "familiarity_seeker"
        ]
    }

    rl_recommender = RLRecommender()
    
    # Initialize Weights & Biases
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "total_timesteps": args.total_timesteps,
                "save_model_path": args.save_model_path,
                "debug": args.debug,
                "num_recommendations": config.get("num_recommendations"),
                "num_candidates": config.get("num_candidates"),
                "catalog_size": config.get("catalog_size"),
                "users_subset": env_params["users_subset"],
            },
            tags=["rl", "recommender", "training"]
        )
    
    rl_recommender.train(
        env_params=env_params,
        num_recommendations=config.get("num_recommendations"),
        total_timesteps=args.total_timesteps,
        debug=args.debug,
        callback=WandbCallback(verbose=0) if not args.no_wandb else None
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
        print("\nRunning comprehensive evaluation for wandb visualization...")
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
    print("Running comprehensive evaluation...")
    
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


