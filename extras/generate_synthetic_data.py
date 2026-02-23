import random
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Countries and categories
countries = ["UK", "FR", "DE", "ES"]
product_categories = ["Electronics", "Home", "Fashion", "Beauty", "Toys"]

# Review templates with intentional "noise" for teaching purposes
# (Mixed casing, emojis, slang, and excess punctuation)
english_reviews = [
    "The product arrived quickly and WORKS PERFECTLY!!! 😍",
    "terrible quality, broke after one day... omg 😡",
    "Good value for money. highly recommend it!",
    "Not what I expected, but still okay i guess. u get what u pay for.",
    "EXCELLENT product, highly recommend! BEST EVER!!!",
    "It's just okay lol. does the job.",
    "i hate this so much, never buying again. TOTAL WASTE OF CASH.",
    "OMG the delivery was so slow but the product is fine i guess...",
    "ppl should know this is smaller than it looks on screen!!",
    "Five stars all the way! ⭐⭐⭐⭐⭐",
]

french_reviews = [
    "Très bon produit, fonctionne parfaitement. MERCI !!!",
    "Le produit est arrivé cassé... Quelle horreur 😡",
    "Qualité raisonnable pour le prix. c'est pas mal.",
    "Pas satisfait, pourrait être mieux... u_u",
    "Excellent rapport qualité prix! JE RECOMMANDE !!!",
    "C'est correct pour le prix. lol",
]

german_reviews = [
    "Tolles Produkt, sehr zufrieden! 😍",
    "Schlechte Qualität, nicht empfohlen. NIE WIEDER !!!",
    "Preis-Leistung ist okay... mMn.",
    "Lieferung war langsam, aber Produkt ist GUT.",
    "Ich liebe dieses Produkt! ⭐⭐⭐",
]

spanish_reviews = [
    "Producto excelente, funciona muy bien. ¡¡¡WOW!!!",
    "Muy mala calidad, no lo recomiendo... fatal 😡",
    "Regular calidad por el precio. estaria mejor si fuera mas barato.",
    "Llegó tarde, pero funciona bien. xd",
    "¡Me encanta este producto! lo mejor que he comprado.",
]


def _pick_text(country: str) -> str:
    if country == "UK":
        return random.choice(english_reviews)
    if country == "FR":
        return random.choice(french_reviews)
    if country == "DE":
        return random.choice(german_reviews)
    return random.choice(spanish_reviews)


def _sentiment_from_rating(rating: int) -> str:
    if rating in (1, 2):
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def generate_reviews(n: int = 1000) -> pd.DataFrame:
    rows = []
    start_date = datetime(2024, 11, 1)

    # Weights to create class imbalance (Positive > Negative)
    # 1: 10%, 2: 10%, 3: 20%, 4: 30%, 5: 30%
    rating_options = [1, 2, 3, 4, 5]
    rating_weights = [0.1, 0.1, 0.2, 0.3, 0.3]

    for i in range(1, n + 1):
        country = random.choice(countries)
        category = random.choice(product_categories)
        text = _pick_text(country)
        
        # Weighted random choice for rating to ensure imbalance (more positive reviews)
        rating = random.choices(rating_options, weights=rating_weights, k=1)[0]
        sentiment = _sentiment_from_rating(rating)
        ts = start_date + timedelta(days=random.randint(0, 30))

        rows.append(
            [
                i,
                category,
                ts.strftime("%Y-%m-%d"),
                country,
                rating,
                text,
                sentiment,
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "review_id",
            "product_category",
            "timestamp",
            "country",
            "rating",
            "review",
            "sentiment",
        ],
    )

    return df


if __name__ == "__main__":
    # Generate 1000 reviews for a rich dataset
    df = generate_reviews(1000)
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "raw" / "raw_reviews.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Noise-Rich Dataset Created: {out_path}")
    print(f"📊 Sentiment Distribution:\n{df['sentiment'].value_counts()}")
