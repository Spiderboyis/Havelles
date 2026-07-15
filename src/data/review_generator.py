"""
Synthetic Review Data Generator for Havells Consumer Products.

Generates realistic, diverse customer reviews simulating real-world patterns:
- Temporal trends (sentiment shifts over months)
- Product-specific issue patterns
- Varied review lengths and writing styles
- Realistic rating distributions
- Regional language influences (Indian English patterns)
"""

import random
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import uuid

# ─── Review Templates by Category ──────────────────────────────────

REVIEW_TEMPLATES = {
    "fans": {
        "positive": [
            "Bought the {product} {time_ago}. The air delivery is {adj_pos} and it runs {adv_pos}. {extra_pos} Worth every penny!",
            "Excellent fan! The {feature} is top-notch. {extra_pos} Installation was smooth. Happy with the purchase.",
            "Using this {product} for {duration} now. Very satisfied with the performance. {extra_pos} The BLDC motor saves a lot on electricity.",
            "Best ceiling fan I've used. The {feature} makes it stand out. {extra_pos} Recommended for anyone looking for a quality fan.",
            "Premium quality fan from Havells. {extra_pos} Energy efficient and stylish design. Goes well with my room decor.",
            "Got this for my living room and couldn't be happier. The airflow is amazing even at medium speed. {extra_pos}",
            "After trying many brands, finally settled on Havells. The build quality is {adj_pos}. {extra_pos} No regrets.",
        ],
        "negative": [
            "Disappointed with the {product}. The {issue} started within {duration}. {extra_neg} Not expected from Havells.",
            "Fan makes too much {issue}. Called customer care but no proper response. {extra_neg} Very frustrated.",
            "The {feature} stopped working after {duration}. {extra_neg} Service center says it's not covered under warranty. Waste of money.",
            "Wobbling issue from day one. {extra_neg} The technician came but couldn't fix it. Still waiting for replacement.",
            "Bought this fan expecting Havells quality but it's very {adj_neg}. {extra_neg} The motor heats up quickly.",
            "Remote control stopped working in {duration}. Service team is not responding. {extra_neg} Very poor after-sales support.",
            "PCB board failed after {duration}. {extra_neg} Havells service center asking for Rs 1500 for replacement even under warranty.",
        ],
        "mixed": [
            "The fan looks good and air delivery is decent, but the {issue} is a problem. {extra_mix}",
            "Good fan for the price but {issue}. {extra_mix} Would have been 5 stars otherwise.",
            "Design is {adj_pos} but the {feature} could be better. {extra_mix} Okay for the price range.",
            "Air flow is good but makes noise at speed 4 and 5. {extra_mix} Average experience overall.",
        ],
        "features": ["air delivery", "motor quality", "remote control", "speed regulation", "design", "energy efficiency", "LED light", "timer function"],
        "issues": ["wobbling", "noise at high speed", "motor heating", "PCB failure", "remote malfunction", "clicking sound", "slow speed", "vibration"],
    },
    "water_heaters": {
        "positive": [
            "Excellent water heater. Heats water in {time_detail}. {extra_pos} The safety features are {adj_pos}.",
            "Using this geyser for {duration}. No issues at all. {extra_pos} Perfect for Indian winters.",
            "Quick heating and energy efficient. {extra_pos} The temperature control knob works perfectly. Good build quality.",
            "Best geyser in this price range. The {feature} is impressive. {extra_pos} Family is very happy.",
            "Havells geyser performing well for {duration}. {extra_pos} Hot water in 10 minutes even in peak winter.",
            "Compact design, fits easily in bathroom. {extra_pos} Heating element quality is excellent. Very satisfied customer.",
        ],
        "negative": [   
            "Geyser stopped heating after {duration}. {extra_neg} Thermostat failure according to service center.",
            "Leaking from the bottom. {extra_neg} Warranty claim rejected saying it's installation issue. Very disappointed.",
            "The {issue} problem started within weeks. {extra_neg} Customer care is unhelpful. Will never buy Havells again.",
            "Water not heating properly. Takes 30+ minutes for lukewarm water. {extra_neg} Defective heating element perhaps.",
            "Safety valve keeps releasing water. {extra_neg} Technician visited 3 times but problem persists. Waste of money.",
            "Inner tank rusted within {duration}. {extra_neg} For the price paid, this is unacceptable quality.",
        ],
        "mixed": [
            "Heating is good but {issue}. {extra_mix} Could be better for the premium price.",
            "Decent geyser but the {feature} needs improvement. {extra_mix} Okay product overall.",
            "Works well in summer but struggles in winter. {extra_mix} Expected more from Havells brand.",
        ],
        "features": ["heating speed", "temperature control", "safety valve", "glass-lined tank", "digital display", "energy rating", "auto-cutoff"],
        "issues": ["thermostat failure", "leaking", "slow heating", "unusual noise", "rust formation", "pressure valve issue", "element burnout"],
    },
    "mixer_grinders": {
        "positive": [
            "Powerful mixer grinder! Grinds everything to fine paste. {extra_pos} The {feature} is excellent.",
            "Using for {duration}. Handles all Indian cooking needs perfectly. {extra_pos} Motor is very powerful.",
            "Best mixer grinder for the price. {extra_pos} Wet and dry grinding both are {adj_pos}. Highly recommend.",
            "Havells quality at its best. The jars are sturdy and {feature} works great. {extra_pos}",
            "Makes perfect dosa batter and chutney. {extra_pos} The 750W motor handles everything easily.",
            "Love the look and performance. {extra_pos} Easy to clean and the locking mechanism is solid.",
        ],
        "negative": [
            "Motor burned out after {duration}. {extra_neg} Expected better from a brand like Havells.",
            "Jar cracked while grinding. {extra_neg} Very poor quality material used. Dangerous product.",
            "The {issue} is terrible. {extra_neg} Cannot grind properly. Service center says motor needs replacement.",
            "Coupler breaks frequently. {extra_neg} Already replaced twice in {duration}. Bad design.",
            "Overheats after 5 minutes of use. {extra_neg} The safety feature cuts off motor too frequently.",
            "Blade assembly rusted within {duration}. {extra_neg} Quality has gone down significantly in recent models.",
        ],
        "mixed": [
            "Good for light grinding but struggles with hard items. {extra_mix} The {feature} is decent though.",
            "Works fine for daily use but {issue}. {extra_mix} Average mixer for the price.",
            "Design is premium but {issue} is a concern. {extra_mix} Needs improvement in build quality.",
        ],
        "features": ["grinding performance", "motor power", "jar quality", "safety lock", "speed settings", "overload protection", "cord length"],
        "issues": ["motor overheating", "jar leaking", "coupler breaking", "blade quality", "noise level", "vibration", "switch malfunction"],
    },
    "air_purifiers": {
        "positive": [
            "Excellent air purifier! Air quality improved significantly. {extra_pos} The {feature} is very useful.",
            "PM2.5 readings dropped from 200+ to below 30. {extra_pos} Worth the investment for health.",
            "Silent operation and effective purification. {extra_pos} The HEPA filter does a great job.",
            "Noticed significant improvement in allergies after using this. {extra_pos} Real-time display is helpful.",
            "Good coverage for my 300 sq ft room. {extra_pos} The air feels noticeably cleaner.",
        ],
        "negative": [
            "Filter replacement cost is too high. {extra_neg} Rs 3000+ for HEPA filter is {adj_neg}.",
            "Not effective for the claimed area coverage. {extra_neg} PM2.5 barely drops in my room.",
            "Fan stopped working after {duration}. {extra_neg} Service not available in my city.",
            "The {issue} makes it unusable at night. {extra_neg} Defeats the purpose of having one in bedroom.",
            "Display shows wrong readings. {extra_neg} Compared with external monitor and values don't match.",
        ],
        "mixed": [
            "Purification is okay but {issue}. {extra_mix} Good for smaller rooms only.",
            "Decent product but filter life is less than claimed. {extra_mix} Running cost is high.",
        ],
        "features": ["HEPA filter", "PM2.5 display", "auto mode", "sleep mode", "filter life indicator", "child lock", "coverage area"],
        "issues": ["high noise at max speed", "filter cost", "inaccurate readings", "weak airflow", "filter availability", "sensor malfunction"],
    },
    "lighting": {
        "positive": [
            "Bright and energy efficient LED. {extra_pos} Been using for {duration} with no issues.",
            "The light quality is excellent. {extra_pos} No flickering at all. True to claimed lumens.",
            "Best LED bulb I've bought. Color temperature is perfect. {extra_pos} Great value for money.",
            "Smart light works perfectly with the app. {extra_pos} Color changing feature is amazing.",
            "Installed 10 of these in my home. {extra_pos} Consistent brightness and great build quality.",
        ],
        "negative": [
            "LED started flickering after {duration}. {extra_neg} Three bulbs failed in just 6 months.",
            "The {issue} is very annoying. {extra_neg} Brightness drops significantly over time.",
            "Smart bulb loses WiFi connection frequently. {extra_neg} App is buggy and unresponsive.",
            "Bulb gets extremely hot. {extra_neg} Concerned about safety in enclosed fixtures.",
            "Color rendering is poor. {extra_neg} Everything looks yellowish. Not true to advertised specs.",
        ],
        "mixed": [
            "Good brightness but {issue}. {extra_mix} Okay for the price point.",
            "Energy saving is real but the {feature} could be better. {extra_mix} Decent product overall.",
        ],
        "features": ["brightness", "color temperature", "energy saving", "WiFi connectivity", "dimming", "lifespan", "CRI rating"],
        "issues": ["flickering", "overheating", "WiFi drops", "dim output", "color shift", "buzzing sound", "short lifespan"],
    }
}

# ─── Extra Phrases for Realistic Variation ──────────────────────

EXTRA_POSITIVE = [
    "Build quality is premium.", "Very happy with this purchase.", "Recommend to everyone.",
    "Havells has maintained its quality.", "Service team was also helpful during installation.",
    "The packaging was excellent.", "Good value for money.", "My family loves it.",
    "Energy savings are noticeable on the electricity bill.", "Will buy more Havells products.",
    "Installation was done within 2 days of purchase.", "Works as advertised.",
    "Much better than the Chinese alternatives.", "Sturdy construction.",
    "My neighbor also bought one after seeing mine.", "Using it daily without any issues.",
    "Even after months, performance hasn't degraded.", "The warranty gives peace of mind.",
]

EXTRA_NEGATIVE = [
    "Customer care is useless.", "Will not recommend to anyone.", "Complete waste of money.",
    "Havells quality has gone down.", "No response from service center.",
    "Technician never showed up despite 3 complaints.", "Raised complaint on Havells app but no response.",
    "This is clearly a manufacturing defect.", "I've used other Havells products which were much better.",
    "Returned to Amazon. Getting refund.", "Posted on consumer forum as well.",
    "Filed a complaint with consumer court.", "The product looks cheaply made.",
    "Plastic quality is very poor.", "Regret not buying the Crompton/Bajaj alternative.",
    "Going to switch to another brand now.", "Misleading product description on website.",
]

EXTRA_MIXED = [
    "Overall a decent product.", "Has both pros and cons.", "Good for the price but not premium.",
    "Expected more from Havells at this price point.", "It's okay, nothing special.",
    "Would rate 3/5. Room for improvement.", "Works for basic needs.",
    "Not the best but not the worst either.", "Acceptable performance for daily use.",
]

POSITIVE_ADJECTIVES = ["amazing", "excellent", "fantastic", "superb", "outstanding", "great", "wonderful", "impressive"]
NEGATIVE_ADJECTIVES = ["poor", "terrible", "disappointing", "awful", "substandard", "unacceptable", "pathetic", "mediocre"]
POSITIVE_ADVERBS = ["quietly", "smoothly", "efficiently", "perfectly", "silently", "beautifully"]

PRODUCT_MODELS = {
    "fans": [
        "Havells Stealth Air BLDC", "Havells ES-40 Premium", "Havells Leganza 4B",
        "Havells Pacer", "Havells Festiva", "Havells Sprint 400mm", "Havells Ventil Air DX",
        "Havells Nicola", "Havells Carnival", "Havells SS-390 Metallic"
    ],
    "water_heaters": [
        "Havells Instanio 3L", "Havells Monza EC 15L", "Havells Puro Plus 25L",
        "Havells Adonia Spin 25L", "Havells Carlo 5L", "Havells Quatro 15L",
        "Havells Magnatron 15L", "Havells Bello 25L"
    ],
    "mixer_grinders": [
        "Havells Capture 500W", "Havells Marathon 750W", "Havells PowerGrind 1000W",
        "Havells Vitonica 500W", "Havells Momenta 750W", "Havells Sprint Mixer 600W"
    ],
    "air_purifiers": [
        "Havells Freshia AP-46", "Havells Freshia AP-58", "Havells Studio AP-22",
    ],
    "lighting": [
        "Havells Adore LED 9W", "Havells Lumeno LED 15W", "Havells Glow LED 12W",
        "Havells Smart LED WiFi", "Havells LED Panel 18W", "Havells NXT LED Batten 20W"
    ]
}

REVIEWER_NAMES = [
    "Rahul S.", "Priya M.", "Amit K.", "Sneha T.", "Vikram P.", "Ananya R.",
    "Deepak G.", "Kavita J.", "Suresh L.", "Meera D.", "Rajesh V.", "Pooja B.",
    "Manish C.", "Divya N.", "Arun H.", "Sunita W.", "Prakash A.", "Nisha F.",
    "Rohit E.", "Lakshmi I.", "Sanjay U.", "Geeta O.", "Vivek Y.", "Rina Q.",
    "Manoj Z.", "Archana X.", "Karthik S.", "Swati M.", "Nitin K.", "Pallavi T.",
    "Gaurav P.", "Shruti R.", "Ajay G.", "Rekha J.", "Harsh L.", "Bhavna D.",
    "Tushar V.", "Shweta B.", "Raghav C.", "Anjali N.", "Pankaj H.", "Sapna W.",
    "Venkat A.", "Kamini F.", "Dhruv E.", "Tanvi I.", "Mohan U.", "Ritu O.",
    "Arjun Y.", "Seema Q."
]

PLATFORMS = ["Amazon.in", "Flipkart", "Havells.com", "Reliance Digital", "Croma"]


def _generate_temporal_bias(month: int, category: str) -> float:
    """
    Generate a temporal bias that simulates real-world sentiment trends.
    For example: water heater complaints peak in winter, fan complaints peak in summer.
    """
    seasonal_patterns = {
        "fans": {
            # Complaints increase in summer (Apr-Jul) due to high usage
            4: -0.05, 5: -0.10, 6: -0.12, 7: -0.08,
            # Better sentiment in winter (low usage, fewer failures)
            11: 0.05, 12: 0.08, 1: 0.06
        },
        "water_heaters": {
            # Complaints increase in winter due to high usage
            11: -0.08, 12: -0.12, 1: -0.15, 2: -0.10,
            # Better sentiment in summer (low usage)
            5: 0.05, 6: 0.08, 7: 0.06
        },
        "air_purifiers": {
            # Complaints increase during pollution season (Oct-Jan)
            10: -0.05, 11: -0.12, 12: -0.08, 1: -0.06
        }
    }
    return seasonal_patterns.get(category, {}).get(month, 0.0)


def _select_rating(sentiment: str, temporal_bias: float = 0.0) -> int:
    """Generate a realistic rating based on sentiment with temporal influence."""
    base_weights = {
        "positive": {5: 0.55, 4: 0.30, 3: 0.10, 2: 0.03, 1: 0.02},
        "negative": {5: 0.02, 4: 0.05, 3: 0.13, 2: 0.30, 1: 0.50},
        "mixed":    {5: 0.05, 4: 0.20, 3: 0.45, 2: 0.20, 1: 0.10},
    }
    weights = base_weights[sentiment]
    ratings = list(weights.keys())
    probs = list(weights.values())

    # Apply temporal bias
    if temporal_bias < 0:  # Shift towards negative
        probs[4] += abs(temporal_bias) * 0.5  # increase 1-star
        probs[3] += abs(temporal_bias) * 0.3  # increase 2-star
        probs[0] -= abs(temporal_bias) * 0.8  # decrease 5-star
    elif temporal_bias > 0:  # Shift towards positive
        probs[0] += temporal_bias * 0.5
        probs[4] -= temporal_bias * 0.5

    # Normalize
    total = sum(probs)
    probs = [p / total for p in probs]

    return random.choices(ratings, weights=probs, k=1)[0]


def generate_review(category: str, sentiment: str, review_date: datetime) -> Dict:
    """Generate a single synthetic review with realistic content."""
    templates = REVIEW_TEMPLATES[category]
    template = random.choice(templates[sentiment])

    product = random.choice(PRODUCT_MODELS[category])
    feature = random.choice(templates["features"])
    issue = random.choice(templates["issues"]) if "issues" in templates else "quality"

    durations = ["2 weeks", "1 month", "2 months", "3 months", "6 months", "8 months", "1 year", "1.5 years"]
    time_agos = ["last month", "2 months ago", "last week", "recently", "a few weeks back", "during the sale"]
    time_details = ["5 minutes", "8 minutes", "10 minutes", "15 minutes", "just a few minutes"]

    extra_map = {
        "positive": EXTRA_POSITIVE,
        "negative": EXTRA_NEGATIVE,
        "mixed": EXTRA_MIXED
    }

    review_text = template.format(
        product=product,
        feature=feature,
        issue=issue,
        adj_pos=random.choice(POSITIVE_ADJECTIVES),
        adj_neg=random.choice(NEGATIVE_ADJECTIVES),
        adv_pos=random.choice(POSITIVE_ADVERBS),
        duration=random.choice(durations),
        time_ago=random.choice(time_agos),
        time_detail=random.choice(time_details),
        extra_pos=random.choice(EXTRA_POSITIVE),
        extra_neg=random.choice(EXTRA_NEGATIVE),
        extra_mix=random.choice(EXTRA_MIXED),
    )

    temporal_bias = _generate_temporal_bias(review_date.month, category)
    rating = _select_rating(sentiment, temporal_bias)

    return {
        "review_id": str(uuid.uuid4())[:12],
        "product_name": product,
        "product_category": category,
        "review_text": review_text,
        "rating": rating,
        "sentiment_label": sentiment,
        "reviewer_name": random.choice(REVIEWER_NAMES),
        "review_date": review_date.strftime("%Y-%m-%d"),
        "platform": random.choice(PLATFORMS),
        "verified_purchase": random.random() > 0.15,  # 85% verified
        "helpful_votes": random.randint(0, 150) if random.random() > 0.4 else 0,
    }


def generate_dataset(
    num_reviews: int = 2500,
    start_date: str = "2024-01-01",
    end_date: str = "2025-06-30",
    output_path: Optional[Path] = None,
    sentiment_distribution: Optional[Dict] = None
) -> List[Dict]:
    """
    Generate a complete synthetic review dataset for Havells products.
    
    Args:
        num_reviews: Total number of reviews to generate
        start_date: Start date for review period
        end_date: End date for review period
        output_path: Path to save the dataset (CSV and JSON)
        sentiment_distribution: Custom distribution of sentiments per category
    
    Returns:
        List of review dictionaries
    """
    if sentiment_distribution is None:
        sentiment_distribution = {
            "fans":           {"positive": 0.45, "negative": 0.30, "mixed": 0.25},
            "water_heaters":  {"positive": 0.40, "negative": 0.35, "mixed": 0.25},
            "mixer_grinders": {"positive": 0.50, "negative": 0.25, "mixed": 0.25},
            "air_purifiers":  {"positive": 0.35, "negative": 0.35, "mixed": 0.30},
            "lighting":       {"positive": 0.55, "negative": 0.25, "mixed": 0.20},
        }

    # Category distribution (weighted by product popularity)
    category_weights = {
        "fans": 0.30,
        "water_heaters": 0.22,
        "mixer_grinders": 0.20,
        "air_purifiers": 0.12,
        "lighting": 0.16,
    }

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    date_range = (end - start).days

    reviews = []
    categories = list(category_weights.keys())
    cat_probs = list(category_weights.values())

    for _ in range(num_reviews):
        category = random.choices(categories, weights=cat_probs, k=1)[0]
        dist = sentiment_distribution[category]
        sentiment = random.choices(
            list(dist.keys()),
            weights=list(dist.values()),
            k=1
        )[0]
        review_date = start + timedelta(days=random.randint(0, date_range))

        review = generate_review(category, sentiment, review_date)
        reviews.append(review)

    # Sort by date
    reviews.sort(key=lambda x: x["review_date"])

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_path / "havells_reviews.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)

        # Save as CSV
        csv_path = output_path / "havells_reviews.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=reviews[0].keys())
            writer.writeheader()
            writer.writerows(reviews)

        print(f"✅ Generated {len(reviews)} reviews")
        print(f"   JSON: {json_path}")
        print(f"   CSV:  {csv_path}")
        print(f"   Categories: {dict(zip(categories, [sum(1 for r in reviews if r['product_category'] == c) for c in categories]))}")

    return reviews


if __name__ == "__main__":
    from config.settings import DATA_DIR
    reviews = generate_dataset(num_reviews=2500, output_path=DATA_DIR)
    print(f"\nSample review:\n{json.dumps(reviews[0], indent=2)}")
