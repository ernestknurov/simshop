class User:
    def __init__(self, username: str, gender: bool|None=None, birth_date: str|None=None) -> None:
        self.username = username
        self.gender = gender
        self.birth_date = birth_date
        self.utility_threshold = 0.5

    def react(self, items):
        chosen_items = [
            item for item in items if self.utility(item) >= self.utility_threshold
        ]
        return chosen_items
    
    def utility(self, item) -> float:
        return 0.5
    