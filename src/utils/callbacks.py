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
        self.utility_scores = []
        
        # Loss tracking
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.total_losses = []
        self.explained_variances = []
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout to log training losses."""
        # Access the model's logger to get loss information
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            if hasattr(self.model.logger, 'name_to_value'):
                logger_dict = self.model.logger.name_to_value
                
                loss_metrics = {}
                current_step = self.num_timesteps
                
                # Extract loss values if available
                if 'train/policy_gradient_loss' in logger_dict:
                    policy_loss = float(logger_dict['train/policy_gradient_loss'])
                    self.policy_losses.append(policy_loss)
                    # loss_metrics['loss/policy_loss'] = policy_loss
                
                if 'train/value_loss' in logger_dict:
                    value_loss = float(logger_dict['train/value_loss'])
                    self.value_losses.append(value_loss)
                    # loss_metrics['loss/value_loss'] = value_loss
                
                if 'train/entropy_loss' in logger_dict:
                    entropy_loss = float(logger_dict['train/entropy_loss'])
                    self.entropy_losses.append(entropy_loss)
                    loss_metrics['loss/entropy_loss'] = entropy_loss
                
                if 'train/loss' in logger_dict:
                    total_loss = float(logger_dict['train/loss'])
                    self.total_losses.append(total_loss)
                    # loss_metrics['loss/total_loss'] = total_loss
                
                if 'train/explained_variance' in logger_dict:
                    explained_var = float(logger_dict['train/explained_variance'])
                    self.explained_variances.append(explained_var)
                    # loss_metrics['loss/explained_variance'] = explained_var
                
                # Add rolling averages for smoother visualization
                if len(self.policy_losses) >= 10:
                    loss_metrics['loss/policy_loss_ma10'] = np.mean(self.policy_losses[-10:])
                if len(self.value_losses) >= 10:
                    loss_metrics['loss/value_loss_ma10'] = np.mean(self.value_losses[-10:])
                if len(self.total_losses) >= 10:
                    loss_metrics['loss/total_loss_ma10'] = np.mean(self.total_losses[-10:])
                if len(self.explained_variances) >= 10:
                    loss_metrics['loss/explained_variance_ma10'] = np.mean(self.explained_variances[-10:])
                
                # Log to wandb if we have any loss metrics
                if loss_metrics:
                    wandb.log(loss_metrics)
                    
                if self.verbose > 1 and loss_metrics:
                    print(f"Step {current_step} - Losses: {loss_metrics}")

    def _on_step(self) -> bool:
        # Track step-level rewards for all environments
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        
        # Aggregating across all environments
        if isinstance(rewards, (list, np.ndarray)):
            # Log mean reward across all environments
            step_reward = np.mean(rewards)
        else:
            step_reward = float(rewards)
            
        self.step_rewards.append(step_reward)
        self.current_episode_rewards.append(step_reward)
        self.current_episode_steps += 1
        # Then in the logging section:
        wandb.log({
            "training/mean_step_reward_100": np.mean(self.step_rewards[-100:]) if len(self.step_rewards) >= 100 else np.mean(self.step_rewards)
        })

        if infos:
            # Initialize aggregation containers for this step
            all_ctrs = []
            all_btrs = []
            all_page_counts = []
            all_no_click_pages = []
            episode_ended = False
            all_utility_scores = []
            
            # Process all environments
            for env_idx, info in enumerate(infos):
                if isinstance(info, dict):
                    # Collect metrics from this environment
                    if 'click_through_rate' in info:
                        all_ctrs.append(float(info.get('click_through_rate')))
                    if 'buy_through_rate' in info:
                        all_btrs.append(float(info.get('buy_through_rate')))
                    if "utility_scores" in info:
                        all_utility_scores.append(np.mean(info["utility_scores"]))
                    if 'history' in info and isinstance(info['history'], dict):
                        history = info['history']
                        all_page_counts.append(int(history.get('page_count', 0)))
                        all_no_click_pages.append(int(history.get('consecutive_no_click_pages', 0)))
                    
                    # Check for episode completion
                    if 'episode' in info:
                        episode_info = info['episode']
                        episode_reward = float(episode_info['r'])
                        episode_length = int(episode_info['l'])
                        episode_ended = True
        
            # Now aggregate and log metrics from all environments
            env_metrics = {}
            
            # CTR metrics
            if all_ctrs:
                self.ctr.extend(all_ctrs)
                if len(self.ctr) > 1000:
                    env_metrics['env/mean_ctr_1000'] = np.mean(self.ctr[-1000:])
            
            # BTR metrics
            if all_btrs:
                self.btr.extend(all_btrs)
                if len(self.btr) > 1000:
                    env_metrics['env/mean_btr_1000'] = np.mean(self.btr[-1000:])

            if all_utility_scores:
                self.utility_scores.extend(all_utility_scores)
                if len(self.utility_scores) > 1000:
                    env_metrics['env/mean_utility_score_1000'] = np.mean(self.utility_scores[-1000:])
            
            # Page count metrics
            if all_page_counts:
                self.page_count.extend(all_page_counts)
                if len(self.page_count) > 1000:
                    env_metrics['env/mean_page_count_1000'] = np.mean(self.page_count[-1000:])
            
            # No-click metrics
            if all_no_click_pages:
                self.consecutive_no_click_pages.extend(all_no_click_pages)
                if len(self.consecutive_no_click_pages) > 1000:
                    env_metrics['env/mean_consecutive_no_click_pages_1000'] = np.mean(self.consecutive_no_click_pages[-1000:])
            
            # Log the aggregated metrics
            if env_metrics:
                wandb.log(env_metrics)
        
        # Log episode completion
        if episode_ended:
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Reset current episode tracking
            self.current_episode_rewards = []
            self.current_episode_steps = 0

            episode_metrics = {}
            
            # Calculate rolling averages
            if len(self.episode_rewards) >= 1000:
                episode_metrics["episode/mean_reward_1000"] = np.mean(self.episode_rewards[-1000:])
            
            if len(self.episode_lengths) >= 1000:
                episode_metrics["episode/mean_length_1000"] = np.mean(self.episode_lengths[-1000:])
            
            # Log episode metrics to wandb
            wandb.log(episode_metrics)
            
            if self.verbose > 0:
                print(f"Episode {self.episode_count}: Reward={episode_reward:.3f}, Length={episode_length}")
            
        return True