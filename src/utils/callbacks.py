import wandb
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class NaNCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(NaNCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check model parameters for NaN
        for name, param in self.model.policy.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {name}")
                return False
        return True
    
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
                    if "recommended_items" in info:
                        env_metrics['env/len_items_to_show'] = len(info["recommended_items"])
                    
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

class LogProbCallback(BaseCallback):
    """
    Callback for logging the mean (and recording history) of action log-probabilities
    at the end of each rollout.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.mean_log_probs = []

    def _on_step(self) -> bool:
        # This must be implemented, SB3 calls it every environment step.
        # Returning True means “keep the training going”.
        return True

    def _on_rollout_end(self) -> None:
        # Called at the end of each rollout (before update).
        # rollout_buffer.log_probs: Tensor of shape (n_steps * n_envs,)
        log_probs = self.model.rollout_buffer.log_probs
        # Move to CPU and numpy
        lp = log_probs.cpu().numpy() if hasattr(log_probs, "cpu") else np.array(log_probs)
        mean_lp = float(np.mean(lp))
        self.mean_log_probs.append(mean_lp)

        if self.verbose:
            print(f"[LogProbCallback] Mean log_prob this rollout: {mean_lp:.4f}")

    def _on_training_end(self) -> None:
        # At the very end of training, save the history to disk
        try:
            np.save("mean_log_probs.npy", np.array(self.mean_log_probs))
            if self.verbose:
                print(f"[LogProbCallback] Saved mean_log_probs.npy with {len(self.mean_log_probs)} entries.")
        except Exception as e:
            print(f"[LogProbCallback] Failed to save log_probs history: {e}")