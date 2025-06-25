import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from src.env import ShopEnv
from src.policies import TopKMultiInputPolicy



class DebugCallback(BaseCallback):
    """
    Callback for debugging: prints observations, actions, and rewards at each step.
    """
    def _on_step(self) -> bool:
        obs = self.locals.get('new_obs')
        actions = self.locals.get('actions', None)
        if actions is None:
            actions = self.locals.get('action')
        rewards = self.locals.get('rewards')
        print(f"Debug Step - Obs: {obs}, Action: {actions}, Reward: {rewards}")
        return True


class RLRecommender:
    def __init__(self):
        self.model = None

    def train(self, env_params: dict, num_recommendations: int, total_timesteps: int=2048, debug: bool=False):
        print(f"Training model for {total_timesteps} timesteps on environment ShopEnv")
        def make_env(username: str):
            return lambda: Monitor(ShopEnv(
                items=env_params['catalog'], 
                user=env_params['username_to_user'][username]
                ))

        self.vec_env = DummyVecEnv([make_env(username) for username in env_params['users_subset']])
        # self.model = PPO(TopKMultiInputPolicy, self.vec_env, verbose=1, policy_kwargs={"k": num_recommendations},)
        policy_kwargs = {
            "k": num_recommendations,
            # net_arch can be a list of dicts for PPO’s π- and V-heads:
            "net_arch": dict(
                    pi=[512, 512],   # two hidden layers of 256 units for the policy head
                    vf=[512, 512]
                ),   # two hidden layers of 256 units for the value head,
            "activation_fn": nn.ReLU  # or nn.Tanh if you prefer
        }
        self.model = PPO(
            TopKMultiInputPolicy,
            self.vec_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
        )
       # Use DebugCallback to log step-by-step if debug mode enabled
        callback = DebugCallback() if debug else None
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
        self.evaluate(100)

    def load_model(self, model_path: str, **kwargs):
        # print(f"Loading model from {model_path}")
        self.model = PPO.load(model_path, policy=TopKMultiInputPolicy, **kwargs)

    def save_model(self, model_path: str, **kwargs):
        print(f"Saving model to {model_path}")
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        self.model.save(model_path, **kwargs)

    def evaluate(self, num_episodes: int=100, **kwargs) -> tuple:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        if self.vec_env is None:
            raise ValueError("Vectorized environment is not initialized.")
        print(f"Evaluating model on {num_episodes} episodes")
        mean_reward, std_reward = evaluate_policy(self.model, self.vec_env, n_eval_episodes=num_episodes, **kwargs)
        print(f"Mean reward: {mean_reward} +/- {std_reward}")
        return mean_reward, std_reward

    def predict(self, state: dict, num_recommendations: int=10, **kwargs) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        
        action, _ = self.model.predict(state, **kwargs)
        return action

    def recommend(self, state: dict, num_recommendations: int=10) -> np.ndarray:
        return  self.predict(state, num_recommendations)

