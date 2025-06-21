import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.env import ShopEnv
from src.policies import TopKMultiInputPolicy



class RLRecommender:
    def __init__(self):
        self.model = None

    def train(self, env_params: dict, num_recommendations: int, total_timesteps: int=2048):
        print(f"Training model for {total_timesteps} timesteps on environment ShopEnv")
        def make_env(username: str):
            return lambda: Monitor(ShopEnv(
                items=env_params['catalog'], 
                user=env_params['username_to_user'][username]
                ))

        self.vec_env = DummyVecEnv([make_env(username) for username in env_params['users_subset']])
        self.model = PPO(TopKMultiInputPolicy, self.vec_env, verbose=1, policy_kwargs={"k": num_recommendations},)
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        self.evaluate(100)

    def load_model(self, model_path: str, **kwargs):
        print(f"Loading model from {model_path}")
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

