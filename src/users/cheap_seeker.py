from src.users import User
from typing import Any

class CheapSeekerUser(User):
    def __init__(self, username: str, gender: bool|None=None, birth_date: str|None=None) -> None:
        super().__init__(username, gender, birth_date)

    def choose_item(self, items: list[dict]) -> int | str:
        """
        Choose the cheapest item from the list.
        """
        if not items:
            return "No items available"
        
        prices = [float(item['price']) for item in items]
        cheapest_item_id = prices.index(min(prices))
        
        return cheapest_item_id
    
    def utility(self, item) -> float:
        return 0.5