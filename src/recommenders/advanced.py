import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.env import ShopEnv
from src.policies import TopKMultiInputPolicy, EmbeddingItemEncoder


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
        
        policy_kwargs = {
            "k": num_recommendations,
            "features_extractor_class": EmbeddingItemEncoder,
            "features_extractor_kwargs": {"features_dim": 128},
            "share_features_extractor": True, 
            "net_arch": dict(
                pi=[128, 128],  
                vf=[128, 128]
            ),
        }
        
        self.model = PPO(
            TopKMultiInputPolicy,
            self.vec_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            # n_steps=4096,
            # batch_size=256,
            # n_epochs=10,
        )
        if debug:
            # Debug: Print model architecture
            print("\n[DEBUG] Model architecture:")
            print(self.model.policy)
            
            # Debug: Check feature extractor type
            feature_extractor = self.model.policy.features_extractor
            print(f"\n[DEBUG] Feature extractor type: {type(feature_extractor)}")
            print(f"[DEBUG] Is SharedItemEncoder: {isinstance(feature_extractor, EmbeddingItemEncoder)}")
            
            # Debug: Count parameters
            total_params = sum(p.numel() for p in self.model.policy.parameters())
            feature_extractor_params = sum(p.numel() for p in feature_extractor.parameters())
            print(f"[DEBUG] Total policy parameters: {total_params:,}")
            print(f"[DEBUG] Feature extractor parameters: {feature_extractor_params:,}")

        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
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

