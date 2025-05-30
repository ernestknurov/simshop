from src.users import User
import numpy as np

class BrandLoverUser(User):
    def __init__(self, username: str, favorite_colors: list[str], favorite_brands: list[str], gender: bool|None=None, birth_date: str|None=None) -> None:
        
        super().__init__(username, gender, birth_date)
        self.favorite_colors = favorite_colors
        self.favorite_brands = favorite_brands

    def choose_item(self, items: list[dict]) -> int:
        """
        Choose the items of specific brands and colors from the list.
        """
        if not items:
            return -1  # No items available
        
        filtered_items = [
            item for item in items 
            if item['brand'] in self.favorite_brands and 
               item['color'] in self.favorite_colors
        ]
        if not filtered_items:
            return -1 
        
        filtered_item = np.random.choice(filtered_items)
        item_id = items.index(filtered_item)
        return item_id