import torch
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3.ppo.policies import MultiInputPolicy

class TopKDistribution(BernoulliDistribution):
    """
    BernoulliDistribution that always returns a mask with exactly k ones.
    """
    def __init__(self, action_dims: int, k: int):
        super().__init__(action_dims)
        self.k = k

    def _topk_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Build a {0,1} mask with 1s on the k largest logits.
        logits: (batch, n_actions)
        """
        k = min(self.k, logits.shape[-1])                # guard k > n_actions
        topk_idx = torch.topk(logits, k, dim=-1).indices    # (batch, k)
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask

    # --- the 3 places SB3 may ask for actions -------------------------------
    def actions_from_params(self, action_logits: torch.Tensor,
                            deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(action_logits)           # sets self.distribution
        return self._topk_mask(self.distribution.logits) # ignore `deterministic`

    def sample(self) -> torch.Tensor:
        # SB3's rollout uses Distribution.get_actions() → self.sample()
        return self._topk_mask(self.distribution.logits)

    def mode(self) -> torch.Tensor:
        # used when deterministic=True
        return self._topk_mask(self.distribution.logits)


class TopKMultiInputPolicy(MultiInputPolicy):
    """
    Drop-in replacement for 'MultiInputPolicy' that wires the TopKDistribution
    into PPO/A2C/etc.
    """
    def __init__(self, *args, k: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        # Replace the default BernoulliDistribution with ours
        self.action_dist = TopKDistribution(self.action_space.n, k=self.k)

    def _get_action_dist_from_latent(self, latent_pi):
        """
        Called by SB3 every time it needs a distribution object.
        """
        action_logits = self.action_net(latent_pi)
        # TopKDistribution.proba_distribution() returns *self*, so we return it.
        return self.action_dist.proba_distribution(action_logits)
