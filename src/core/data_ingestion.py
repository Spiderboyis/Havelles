"""
Data Ingestion and Preprocessing Pipeline.

Handles:
- Loading reviews from multiple formats (CSV, JSON)
- Text cleaning and normalization
- Feature extraction (aspects, entities)
- Data validation and deduplication
"""

import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib


@dataclass
class ProcessedReview:
    """Structured representation of a processed review."""
    review_id: str
    product_name: str
    product_category: str
    review_text: str
    cleaned_text: str
    rating: int
    reviewer_name: str
    review_date: str
    platform: str
    verified_purchase: bool
    helpful_votes: int
    word_count: int = 0
    char_count: int = 0
    has_complaint: bool = False
    has_praise: bool = False
    mentioned_aspects: List[str] = field(default_factory=list)
    text_hash: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class DataIngestionPipeline:
    """
    Ingests raw review data, cleans, normalizes, and prepares it for analysis.
    
    Design Decision: This is a stateless pipeline component. Each method takes
    input and produces output without side effects, making it easy to test,
    compose, and swap out individual steps.
    """

    # Common complaint indicators
    COMPLAINT_KEYWORDS = [
        "worst", "terrible", "waste", "defective", "broken", "not working",
        "stopped working", "disappointed", "useless", "regret", "refund",
        "complaint", "frustrated", "pathetic", "cheated", "poor quality",
        "don't buy", "never buy", "avoid", "faulty", "damaged", "fraud"
    ]

    # Common praise indicators
    PRAISE_KEYWORDS = [
        "excellent", "amazing", "best", "love", "perfect", "superb",
        "outstanding", "fantastic", "highly recommend", "worth", "great",
        "premium", "wonderful", "impressive", "happy", "satisfied", "awesome"
    ]

    # Aspect keywords mapping
    ASPECT_KEYWORDS = {
        "performance": ["performance", "speed", "power", "output", "delivery", "works", "functioning"],
        "build_quality": ["quality", "build", "material", "sturdy", "durable", "construction", "premium"],
        "noise": ["noise", "noisy", "sound", "loud", "quiet", "silent", "buzzing", "humming"],
        "service": ["service", "support", "customer care", "technician", "warranty", "repair", "complaint"],
        "installation": ["install", "installation", "setup", "mounting", "fitting"],
        "energy_efficiency": ["energy", "electricity", "bill", "power consumption", "efficient", "savings", "BLDC"],
        "design": ["design", "look", "style", "aesthetic", "color", "finish", "decor"],
        "safety": ["safety", "safe", "shock", "leaking", "overheating", "spark"],
        "value_for_money": ["value", "price", "worth", "expensive", "cheap", "affordable", "cost"],
        "durability": ["durable", "durability", "lasting", "lifespan", "long-term", "years"],
    }

    def load_reviews(self, file_path: Path) -> List[Dict]:
        """Load reviews from CSV or JSON file."""
        file_path = Path(file_path)

        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif file_path.suffix == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize review text.
        
        Steps:
        1. Normalize whitespace and remove extra spaces
        2. Fix common encoding issues
        3. Remove URLs and email addresses
        4. Normalize punctuation
        5. Preserve case for sentiment cues (e.g., "TERRIBLE")
        """
        if not text or not isinstance(text, str):
            return ""

        # Normalize Unicode
        text = text.encode("ascii", errors="ignore").decode("ascii")

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Remove excessive punctuation but keep meaningful ones
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[.]{3,}", "...", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E]", "", text)

        return text

    def extract_aspects(self, text: str) -> List[str]:
        """Extract mentioned product aspects from review text."""
        text_lower = text.lower()
        found_aspects = []

        for aspect, keywords in self.ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found_aspects.append(aspect)
                    break  # One match per aspect is sufficient

        return found_aspects

    def detect_sentiment_signals(self, text: str) -> Tuple[bool, bool]:
        """Detect explicit complaint and praise signals in text."""
        text_lower = text.lower()

        has_complaint = any(kw in text_lower for kw in self.COMPLAINT_KEYWORDS)
        has_praise = any(kw in text_lower for kw in self.PRAISE_KEYWORDS)

        return has_complaint, has_praise

    def compute_text_hash(self, text: str) -> str:
        """Compute hash for deduplication."""
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def process_single_review(self, raw_review: Dict) -> ProcessedReview:
        """Process a single raw review into a structured format."""
        review_text = raw_review.get("review_text", "")
        cleaned = self.clean_text(review_text)
        aspects = self.extract_aspects(cleaned)
        has_complaint, has_praise = self.detect_sentiment_signals(cleaned)
        text_hash = self.compute_text_hash(cleaned)

        return ProcessedReview(
            review_id=raw_review.get("review_id", ""),
            product_name=raw_review.get("product_name", ""),
            product_category=raw_review.get("product_category", ""),
            review_text=review_text,
            cleaned_text=cleaned,
            rating=int(raw_review.get("rating", 0)),
            reviewer_name=raw_review.get("reviewer_name", ""),
            review_date=raw_review.get("review_date", ""),
            platform=raw_review.get("platform", ""),
            verified_purchase=str(raw_review.get("verified_purchase", "True")).lower() == "true",
            helpful_votes=int(raw_review.get("helpful_votes", 0)),
            word_count=len(cleaned.split()),
            char_count=len(cleaned),
            has_complaint=has_complaint,
            has_praise=has_praise,
            mentioned_aspects=aspects,
            text_hash=text_hash,
        )

    def process_batch(self, raw_reviews: List[Dict], deduplicate: bool = True) -> List[ProcessedReview]:
        """
        Process a batch of raw reviews.
        
        Args:
            raw_reviews: List of raw review dictionaries
            deduplicate: Whether to remove duplicate reviews
            
        Returns:
            List of processed, deduplicated reviews
        """
        processed = []
        seen_hashes = set()

        for raw in raw_reviews:
            review = self.process_single_review(raw)

            # Skip empty reviews
            if not review.cleaned_text or review.word_count < 3:
                continue

            # Deduplicate
            if deduplicate:
                if review.text_hash in seen_hashes:
                    continue
                seen_hashes.add(review.text_hash)

            processed.append(review)

        return processed

    def get_ingestion_stats(self, processed: List[ProcessedReview]) -> Dict:
        """Generate statistics about the ingested data."""
        if not processed:
            return {"total": 0}

        categories = {}
        platforms = {}
        ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        complaint_count = 0
        praise_count = 0
        total_words = 0

        for r in processed:
            categories[r.product_category] = categories.get(r.product_category, 0) + 1
            platforms[r.platform] = platforms.get(r.platform, 0) + 1
            ratings[r.rating] = ratings.get(r.rating, 0) + 1
            if r.has_complaint:
                complaint_count += 1
            if r.has_praise:
                praise_count += 1
            total_words += r.word_count

        return {
            "total_reviews": len(processed),
            "categories": categories,
            "platforms": platforms,
            "rating_distribution": ratings,
            "avg_rating": sum(r.rating for r in processed) / len(processed),
            "complaint_percentage": round(complaint_count / len(processed) * 100, 1),
            "praise_percentage": round(praise_count / len(processed) * 100, 1),
            "avg_word_count": round(total_words / len(processed), 1),
            "date_range": {
                "earliest": min(r.review_date for r in processed),
                "latest": max(r.review_date for r in processed),
            },
            "verified_purchase_pct": round(
                sum(1 for r in processed if r.verified_purchase) / len(processed) * 100, 1
            ),
        }
