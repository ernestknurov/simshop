from typing import Any
class Config:
    def __init__(self):
        BRAND_WEIGHTS = {
            'BrandA': 0.5729550570477008,
            'BrandB': 0.004980885074884081,
            'BrandC': 1.516876189441327e-10,
            'BrandD': 0.260225277118719,
            'BrandE': 0.0738844694857421,
            'BrandF': 0.05370536195408048,
            'BrandG': 0.02256994679147764,
            'BrandH': 0.07540303062942973,
            'BrandI': 0.3251875774469237,
            'BrandJ': 0.001179427675238534,
            'BrandK': 0.0014212710996049664,
            'BrandL': 0.10840986357263528,
            'BrandM': 0.0013818673721953775,
            'BrandN': 2.098993484748978e-05,
            'BrandO': 0.2941492051729788
            }
        COLOR_WEIGHTS = {
            'White': 0.09470874002848481,
            'Black': 0.14567775022972448,
            'Red': 0.017433425579196523,
            'Blue': 0.37628387578227307,
            'Green': 0.6653733822187077,
            'Yellow': 0.006109355805172206
        }
        self._config = {
            "user_params": {
                "cheap_seeker": {"click_threshold": 0.75, "buy_threshold": 0.85, "pivot_price": 40},
                "brand_lover": {"click_threshold": 0.30, "buy_threshold": 0.50, "brand_weights": BRAND_WEIGHTS, "color_weights": COLOR_WEIGHTS},
                "value_optimizer": {"click_threshold": 0.7, "buy_threshold": 0.85},
                "familiarity_seeker": {"click_threshold": 0.6, "buy_threshold": 0.80},
                "random_chooser": {"click_threshold": 0.85, "buy_threshold": 0.97},
                "freshness_looker": {"click_threshold": 0.40, "buy_threshold": 0.65, "decay_rate": 0.005},
            },
            "users_list": [
                "cheap_seeker",
                "brand_lover",
                "value_optimizer",
                "familiarity_seeker",
                "random_chooser",
                "freshness_looker"
            ],
            "catalog": {
                "cat_features": ["category", "subcategory", "brand", "color"],
                "num_features": ['price', 'quality_score', 'popularity', 'days_since_release'],
            },
            "models": {
                "model_path": "models/rl_recommender.zip",
            },
            "num_candidates": 50,
            "num_recommendations": 10,
            "catalog_size": 250,
            "n_last_clicks": 5,
            "catalog_path": "data/catalog.csv",
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(
                pi=[256, 256],  
                vf=[256, 256]
            ),
        }

    def get(self, key: str, default: Any=None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        self._config[key] = value