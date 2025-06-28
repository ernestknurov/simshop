import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import BernoulliDistribution
from stable_baselines3.ppo.policies import MultiInputPolicy
from gymnasium.spaces import Dict

class EmbeddingItemEncoder(BaseFeaturesExtractor):
    """
    Feature extractor using embeddings for categorical features instead of one-hot.
    This dramatically reduces parameters: 4 categorical features → 4 small embeddings.
    """
    def __init__(self, observation_space: Dict, features_dim: int = 256, embed_dim: int = 8):
        super().__init__(observation_space, features_dim)
        
        # Get dimensions from observation space
        self.num_candidates = observation_space["candidates_cat_features"].shape[0]
        self.num_features_dim = observation_space["candidates_num_features"].shape[1]
        self.user_dim = observation_space["user"].shape[0]
        self.embed_dim = embed_dim
        
        # Vocabulary sizes for each categorical feature (from your catalog)
        self.vocab_sizes = {
            'category': 7,      # Electronics, Clothing, Home, Sports, Beauty, Books, Toys
            'subcategory': 21,  # Approximate from your data
            'brand': 15,        # BrandA through BrandO  
            'color': 6          # White, Black, Red, Blue, Green, Yellow
        }
        
        # Embedding layers for categorical features
        self.category_embed = nn.Embedding(self.vocab_sizes['category'], embed_dim)
        self.subcategory_embed = nn.Embedding(self.vocab_sizes['subcategory'], embed_dim)
        self.brand_embed = nn.Embedding(self.vocab_sizes['brand'], embed_dim)
        self.color_embed = nn.Embedding(self.vocab_sizes['color'], embed_dim)
        
        # Item encoder - now processes embeddings + numerical features
        item_input_dim = 4 * embed_dim + self.num_features_dim  # 4 embeddings + numerical
        self.item_encoder = nn.Sequential(
            nn.Linear(item_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # User encoder (unchanged)
        self.user_encoder = nn.Sequential(
            nn.Linear(self.user_dim, 32),
            # nn.ReLU(),
            # nn.Linear(32, 16)
        )
        
        # User projection for attention (16 → 32)
        # self.user_projection = nn.Linear(16, 32)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        
        # Feature combiner
        self.feature_combiner = nn.Sequential(
            nn.Linear(32 + 32 + 32, features_dim),  # user + attended_items + item_stats
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: dict) -> torch.Tensor:
        batch_size = observations["user"].shape[0]
        
        # Process user features
        user_features = self.user_encoder(observations["user"])  # (batch, 16)
        
        # Get categorical indices (assuming they come as indices, not one-hot)
        cat_indices = observations["candidates_cat_features"].long()  # (batch, num_candidates, cat_dim)
        num_features = observations["candidates_num_features"]  # (batch, num_candidates, num_dim)
        
        # Apply embeddings to each categorical feature
        category_emb = self.category_embed(cat_indices[:, :, 0])    # (batch, num_candidates, embed_dim)
        subcategory_emb = self.subcategory_embed(cat_indices[:, :, 1])
        brand_emb = self.brand_embed(cat_indices[:, :, 2])
        color_emb = self.color_embed(cat_indices[:, :, 3])
        
        # Concatenate all embeddings + numerical features
        item_features = torch.cat([
            category_emb, subcategory_emb, brand_emb, color_emb, num_features
        ], dim=-1)  # (batch, num_candidates, 4*embed_dim + num_features)
        
        # Apply shared encoder to all items
        batch_items = item_features.view(-1, 4 * self.embed_dim + self.num_features_dim)
        encoded_items = self.item_encoder(batch_items)  # (batch * num_candidates, 32)
        encoded_items = encoded_items.view(batch_size, self.num_candidates, 32)
        
        # User-item attention - project user features to match attention dimension
        # user_query = self.user_projection(user_features).unsqueeze(1)  # (batch, 1, 32)
        user_query = user_features.unsqueeze(1)  # (batch, 1, 32)
        
        attended_items, _ = self.attention(user_query, encoded_items, encoded_items)
        attended_items = attended_items.squeeze(1)  # (batch, 32)
        
        # Aggregate item statistics
        item_stats = torch.mean(encoded_items, dim=1)  # (batch, 32)
        
        # Combine all features
        combined_features = torch.cat([user_features, attended_items, item_stats], dim=-1)
        
        return self.feature_combiner(combined_features)

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
