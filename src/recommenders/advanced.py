import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from src.env import ShopEnv
from src.models.policies import TopKMultiInputPolicy, EmbeddingItemEncoder
from src.utils.logger import get_logger
from src.config import Config

import torch

logger = get_logger(__name__, level="DEBUG")
config = Config()

class RLRecommender:
    def __init__(self):
        self.model = None

    def train(self, env_params: dict, num_recommendations: int, total_timesteps: int=2048, callback=None):
        print(f"Training model for {total_timesteps} timesteps on environment ShopEnv")
        def make_env(username: str):
            return lambda: Monitor(ShopEnv(
                items=env_params['catalog'], 
                user=env_params['username_to_user'][username]
                ))

        self.vec_env = SubprocVecEnv([make_env(username) for username in env_params['users_subset']*8]) 
        # self.vec_env = VecNormalize(self.vec_env, norm_obs=True, norm_reward=True)
        
        policy_kwargs = {
            "k": num_recommendations,
            "features_extractor_class": EmbeddingItemEncoder,
            "features_extractor_kwargs": config.get("features_extractor_kwargs"),
            "share_features_extractor": True, 
            "net_arch": config.get("net_arch"),
        }
        
        self.model = PPO(
            TopKMultiInputPolicy,
            # "MultiInputPolicy",
            self.vec_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
                # === rollout / batch settings ===
            n_steps=512,          # 512×8 = 4 096 samples per update
            batch_size=256,       # 4 096/256 = 16 minibatches per epoch
            n_epochs=10,          # 16×10 = 160 gradient steps per update

            # === optimization ===
            learning_rate=3e-4,   # default “3e-4” works well for ~10⁵ parameters
            clip_range=0.2,       # PPO clipping ε
            gamma=0.99,           # discount factor
            gae_lambda=0.95,      # GAE smoothing

            # === losses / regularization ===
            ent_coef=0.0,         # you can raise to ~1e-2 if you need more exploration
            vf_coef=0.5,          # value function loss weight
            max_grad_norm=0.5     # gradient clipping
            
        )

        # Debug: Print model architecture
        logger.debug("Model architecture:")
        logger.debug(self.model.policy)
        
        # Debug: Check feature extractor type
        feature_extractor = self.model.policy.features_extractor
        logger.debug(f"\nFeature extractor type: {type(feature_extractor)}")
        logger.debug(f"Is SharedItemEncoder: {isinstance(feature_extractor, EmbeddingItemEncoder)}")
        
        # Debug: Count parameters
        total_params = sum(p.numel() for p in self.model.policy.parameters())
        feature_extractor_params = sum(p.numel() for p in feature_extractor.parameters())
        logger.debug(f"Total policy parameters: {total_params:,}")
        logger.debug(f"Feature extractor parameters: {feature_extractor_params:,}")

        self.model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callback)
        self.evaluate(100)

    def load_model(self, model_path: str, **kwargs):
        # print(f"Loading model from {model_path}")
        self.model = PPO.load(model_path, policy=TopKMultiInputPolicy, **kwargs)
        # self.model = PPO.load(model_path, policy="MultiInputPolicy", **kwargs)

    def save_model(self, model_path: str, **kwargs):
        logger.info(f"Saving model to {model_path}")
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        self.model.save(model_path, **kwargs)

    def evaluate(self, num_episodes: int=100, **kwargs) -> tuple:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        if self.vec_env is None:
            raise ValueError("Vectorized environment is not initialized.")
        logger.debug(f"Evaluating model on {num_episodes} episodes")
        mean_reward, std_reward = evaluate_policy(self.model, self.vec_env, n_eval_episodes=num_episodes, **kwargs)
        logger.debug(f"Mean reward: {mean_reward} +/- {std_reward}")
        return mean_reward, std_reward

    def predict(self, state: dict, num_recommendations: int=10, **kwargs) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        
        action, _ = self.model.predict(state, **kwargs)
        return action

    def recommend(self, state: dict, num_recommendations: int=10) -> np.ndarray:
        return  self.predict(state, num_recommendations)

