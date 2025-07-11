import os
import csv
import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__, level="INFO")

fake = Faker()

CATEGORIES = {
    "Electronics": ["Smartphone", "Laptop", "Headphones"],
    "Clothing": ["T-Shirt", "Jeans", "Jacket"],
    "Home": ["Chair", "Lamp", "Table"],
    "Sports": ["Basketball", "Tennis Racket", "Yoga Mat"],
    "Beauty": ["Lipstick", "Perfume", "Cream"],
    "Books": ["Novel", "Cookbook", "Biography"],
    "Toys": ["Puzzle", "Doll", "Board Game"]
}
BRANDS = [f"Brand{chr(i)}" for i in range(65, 80)]  # A–O
COLORS = ["White", "Black", "Red", "Blue", "Green", "Yellow"]

def generate_catalog(n_items=250):
    rows = []
    start_date = datetime.now() - timedelta(days=730)
    for pid in range(1, n_items+1):
        cat = random.choice(list(CATEGORIES.keys()))
        sub = random.choice(CATEGORIES[cat])
        price = float(np.random.lognormal(mean=3.5, sigma=0.75))
        quality = round(random.random(), 3)
        # Popularity as Poisson: lambda = 20 + 80*quality
        pop = np.random.poisson(lam=20 + 80*quality)
        release = start_date + timedelta(days=random.randint(0, 730))
        rows.append({
            "product_id": pid,
            "name": f"{sub} {fake.lexify(text='??##').upper()}",
            "category": cat,
            "subcategory": sub,
            "price": round(price, 2),
            "quality_score": quality,
            "brand": random.choice(BRANDS),
            "color": random.choice(COLORS),
            "popularity": int(pop),
            "release_date": release.date().isoformat(),
            "description": fake.sentence(nb_words=12)
        })
    return rows

def save_to_csv(rows, path="data/catalog.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    catalog = generate_catalog()
    save_to_csv(catalog, "data/catalog.csv")
    logger.info(f"Catalog generated: {len(catalog)} items → data/catalog.csv")
