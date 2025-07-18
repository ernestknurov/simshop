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
        self.num_clicked_items = observation_space['history_n_last_click_items_num_features'].shape[0]
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
        self.item_input_dim = 4 * embed_dim + self.num_features_dim  # 4 embeddings + numerical
        self.item_encoder = nn.Sequential(
            nn.Linear(self.item_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # User encoder (unchanged)
        self.user_encoder = nn.Sequential(
            nn.Linear(self.user_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        
        # Feature combiner
        self.feature_combiner = nn.Sequential(
            nn.Linear(32 + 32 + 32 + 32, features_dim),  # user + attended_candidates_stats + candidates_stats + history_stats
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
    
    def forward(self, observations: dict) -> torch.Tensor:
        batch_size = observations["user"].shape[0]
        
        # Process user features
        user_features = self.user_encoder(observations["user"])  # (batch, 32)
        
        # Get categorical indices (assuming they come as indices, not one-hot)
        candidates_cat_features = observations["candidates_cat_features"].long()  # (batch, num_candidates, cat_dim)
        candidates_num_features = observations["candidates_num_features"]  # (batch, num_candidates, num_dim)
        clicked_items_cat_features = observations['history_n_last_click_items_cat_features'].long() # (batch, num_clicked_items, cat_dim)
        clicked_items_num_features = observations['history_n_last_click_items_num_features'] # (batch, num_clicked_items, num_dim)
        clicked_items_mask = observations['history_n_last_click_items_mask'].float() # (batch, num_clicked_items)
        
        # ------------apply items transformations for candidates items----------------------------------------------------------------

        # Apply embeddings to each categorical feature
        category_emb = self.category_embed(candidates_cat_features[:, :, 0])    # (batch, num_candidates, embed_dim)
        subcategory_emb = self.subcategory_embed(candidates_cat_features[:, :, 1])
        brand_emb = self.brand_embed(candidates_cat_features[:, :, 2])
        color_emb = self.color_embed(candidates_cat_features[:, :, 3])
        
        # Concatenate all embeddings + numerical features
        candidates_item_features = torch.cat([
            category_emb, subcategory_emb, brand_emb, color_emb, candidates_num_features
        ], dim=-1)  # (batch, num_candidates, 4*embed_dim + num_features)
        
        # Apply shared encoder to all items
        candidates_batch_items = candidates_item_features.view(-1, self.item_input_dim)
        encoded_candidates_items = self.item_encoder(candidates_batch_items)  # (batch * num_candidates, 32)
        encoded_candidates_items = encoded_candidates_items.view(batch_size, self.num_candidates, 32)

        # ------------apply items transformations for user preference items----------------------------------------------------------------

        # Apply embeddings to each categorical feature
        category_emb = self.category_embed(clicked_items_cat_features[:, :, 0])    # (batch, clicked_items, embed_dim)
        subcategory_emb = self.subcategory_embed(clicked_items_cat_features[:, :, 1])
        brand_emb = self.brand_embed(clicked_items_cat_features[:, :, 2])
        color_emb = self.color_embed(clicked_items_cat_features[:, :, 3])
        
        # Concatenate all embeddings + numerical features + 1 (mask)
        clicked_item_features = torch.cat([
            category_emb, subcategory_emb, brand_emb, color_emb, clicked_items_num_features
        ], dim=-1)  # (batch, num_clicked_items, 4*embed_dim + num_features)
        
        # Apply shared encoder to all items
        clicked_batch_items = clicked_item_features.view(-1, self.item_input_dim)
        encoded_clicked_items = self.item_encoder(clicked_batch_items)  # (batch * num_clicked_items, 32)
        encoded_clicked_items = encoded_clicked_items.view(batch_size, self.num_clicked_items, 32)
        
        # ----------------------------------------------------------------------------
        # encoded_candidates_items: (batch, num_candidates, 32)
        # encoded_clicked_items:   (batch, num_clicked_items, 32)
        # clicked_items_mask:      (batch, num_clicked_items)

        pad_mask = clicked_items_mask.eq(0)    # (batch, num_clicked_items)

        all_sequences_masked = pad_mask.all(dim=1)  # Check each batch item)
        
        # Handle cold start: if no history, skip attention and use candidates as fallback
        if all_sequences_masked.any():
            print(f"[DEBUG] Cold start detected - using candidates_item_stats as fallback for attention")
            # For cold start users, use candidates statistics instead of attention
            attended_candidates = encoded_candidates_items.clone()
        else:
            # for given history of clicked items, which candidates to attend to?
            attended_candidates, _ = self.attention(
                query=encoded_candidates_items,                # (batch, num_candidates, 32)
                key=encoded_clicked_items,                     # (batch, num_clicked_items, 32)
                value=encoded_clicked_items,                   # (batch, num_clicked_items, 32)
                key_padding_mask=pad_mask                      # (batch, num_clicked_items)
            )
        # → attended_cands: (batch, num_candidates, 32)
        
        # Aggregate item statistics - apply mask to avoid including padded items
        attended_candidates_stats = torch.mean(attended_candidates, dim=1)  # (batch, 32)
        candidates_item_stats = torch.mean(encoded_candidates_items, dim=1)  # (batch, 32)
        
        # For clicked items, mask out padded items before computing mean
        valid_mask = clicked_items_mask.unsqueeze(-1)  # (batch, num_clicked_items, 1)
        masked_clicked_items = encoded_clicked_items * valid_mask  # Zero out padded items
        
        # Compute mean only over valid items
        sum_clicked_items = torch.sum(masked_clicked_items, dim=1)  # (batch, 32)
        count_valid_items = torch.sum(clicked_items_mask, dim=1, keepdim=True)  # (batch, 1)
        count_valid_items = torch.clamp(count_valid_items, min=1)  # Avoid division by zero
        clicked_item_stats = sum_clicked_items / count_valid_items  # (batch, 32)
        
        # Combine all features
        combined_features = torch.cat([
            user_features, 
            attended_candidates_stats, 
            candidates_item_stats, 
            clicked_item_stats
            ], dim=-1)
        
        return self.feature_combiner(combined_features)

# class TopKDistribution(BernoulliDistribution):
#     """
#     BernoulliDistribution that always returns a mask with exactly k ones.
#     """
#     def __init__(self, action_dims: int, k: int):
#         super().__init__(action_dims)
#         self.k = k

#     def _topk_mask(self, logits: torch.Tensor) -> torch.Tensor:
#         """
#         Build a {0,1} mask with 1s on the k largest logits.
#         logits: (batch, n_actions)
#         """
#         k = min(self.k, logits.shape[-1])                # guard k > n_actions
#         topk_idx = torch.topk(logits, k, dim=-1).indices    # (batch, k)
#         mask = torch.zeros_like(logits)
#         mask.scatter_(-1, topk_idx, 1.0)
#         return mask

#     # --- the 3 places SB3 may ask for actions -------------------------------
#     def actions_from_params(self, action_logits: torch.Tensor,
#                             deterministic: bool = False) -> torch.Tensor:
#         self.proba_distribution(action_logits)           # sets self.distribution
#         return self._topk_mask(self.distribution.logits) # ignore `deterministic`

#     def sample(self) -> torch.Tensor:
#         # SB3's rollout uses Distribution.get_actions() → self.sample()
#         return self._topk_mask(self.distribution.logits)

#     def mode(self) -> torch.Tensor:
#         # used when deterministic=True
#         return self._topk_mask(self.distribution.logits)

class TopKDistribution(BernoulliDistribution):
    """
    Samples k items without replacement and supports policy-gradient updates.
    """
    def __init__(self, action_dims: int, k: int):
        super().__init__(action_dims)
        self.k = k
        self.logits = None
        self.probs = None

    def proba_distribution(self, logits: torch.Tensor) -> "TopKDistribution":
        """
        Store logits and compute softmax probabilities.
        """
        self.logits = logits
        self.probs = torch.softmax(logits, dim=-1)
        return self

    def sample(self) -> torch.Tensor:
        """
        Sample k distinct items (mask of shape [batch, n_actions]).
        """
        idx = torch.multinomial(self.probs, self.k, replacement=False)
        mask = torch.zeros_like(self.probs)
        mask.scatter_(-1, idx, 1.0)
        return mask

    def actions_from_params(self, action_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.proba_distribution(action_logits)
        if deterministic:
            # deterministic top-k
            _, topk_idx = torch.topk(self.logits, self.k, dim=-1)
            mask = torch.zeros_like(self.logits)
            mask.scatter_(-1, topk_idx, 1.0)
            return mask
        else:
            return self.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability of chosen k-hot mask under the factorized policy.
        actions: mask tensor [batch, n_actions]
        """
        # Extract indices of the chosen actions via topk on the mask
        chosen_idx = actions.topk(self.k, dim=-1).indices  # [batch, k]
        chosen_p = self.probs.gather(-1, chosen_idx)        # [batch, k]
        return torch.log(chosen_p + 1e-8).sum(-1)           # [batch]

    def entropy(self) -> torch.Tensor:
        """
        Approximate entropy as sum of individual categorical entropies.
        """
        return -(self.probs * torch.log(self.probs + 1e-8)).sum(-1)

    def mode(self) -> torch.Tensor:
        # deterministic top-k
        _, topk_idx = torch.topk(self.logits, self.k, dim=-1)
        mask = torch.zeros_like(self.logits)
        mask.scatter_(-1, topk_idx, 1.0)
        return mask

class TopKMultiInputPolicy(MultiInputPolicy):
    """
    Policy that uses TopKDistribution for slate selection.
    """
    def __init__(self, *args, k: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        # initialize our custom distribution with dims and k
        self.action_dist = TopKDistribution(self.action_space.n, k=self.k)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):
        """
        Build the TopKDistribution from the policy network logits.
        """
        logits = self.action_net(latent_pi)  # [batch, n_actions]
        return self.action_dist.proba_distribution(logits)

# class TopKMultiInputPolicy(MultiInputPolicy):
#     """
#     Drop-in replacement for 'MultiInputPolicy' that wires the TopKDistribution
#     into PPO/A2C/etc.
#     """
#     def __init__(self, *args, k: int = 10, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.k = k
#         # Replace the default BernoulliDistribution with ours
#         self.action_dist = TopKDistribution(self.action_space.n, k=self.k)

#     def _get_action_dist_from_latent(self, latent_pi):
#         """
#         Called by SB3 every time it needs a distribution object.
#         """
#         action_logits = self.action_net(latent_pi)
#         # TopKDistribution.proba_distribution() returns *self*, so we return it.
#         return self.action_dist.proba_distribution(action_logits)
