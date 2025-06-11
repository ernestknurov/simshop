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
            }
        }

    def get(self, key):
        return self._config.get(key)

    def set(self, key, value):
        self._config[key] = value