"""
Emoji mappings for product categories and subcategories.
"""

# Category to emoji mapping
CATEGORY_EMOJI_MAP = {
    # Home category
    "Home": {
        "Chair": "🪑",
        "Table": "🪑", 
        "Lamp": "💡",
        "default": "🏠"
    },
    # Beauty category
    "Beauty": {
        "Perfume": "🌸",
        "Cream": "🧴",
        "Lipstick": "💄",
        "default": "💅"
    },
    # Sports category
    "Sports": {
        "Tennis Racket": "🎾",
        "Basketball": "🏀",
        "Yoga Mat": "🧘",
        "default": "⚽"
    },
    # Books category
    "Books": {
        "Novel": "📚",
        "Cookbook": "👩‍🍳",
        "Biography": "📖",
        "default": "📙"
    },
    # Clothing category
    "Clothing": {
        "T-Shirt": "👕",
        "Jeans": "👖",
        "Jacket": "🧥",
        "default": "👔"
    },
    # Electronics category
    "Electronics": {
        "Smartphone": "📱",
        "Laptop": "💻",
        "Headphones": "🎧",
        "default": "📺"
    },
    # Toys category
    "Toys": {
        "Doll": "🪆",
        "Puzzle": "🧩",
        "Board Game": "🎲",
        "default": "🧸"
    }
}


def get_emoji_for_product(category: str, subcategory: str) -> str:
    """
    Get emoji based on category and subcategory.
    
    Args:
        category: Product category (e.g., "Home", "Beauty", "Sports")
        subcategory: Product subcategory (e.g., "Chair", "Perfume", "Tennis Racket")
        
    Returns:
        str: Emoji representing the product type
    """
    if category in CATEGORY_EMOJI_MAP:
        category_map = CATEGORY_EMOJI_MAP[category]
        return category_map.get(subcategory, category_map["default"])
    return "📦"  # Default fallback emoji