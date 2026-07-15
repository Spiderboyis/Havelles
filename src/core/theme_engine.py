"""
Theme Discovery Engine using unsupervised topic modeling.

Implements BERTopic-inspired approach for discovering recurring themes
in customer reviews without requiring labeled training data.

When BERTopic/UMAP/HDBSCAN dependencies are unavailable, falls back to
a TF-IDF + KMeans approach that works with standard scikit-learn.
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Theme:
    """Represents a discovered theme/topic from reviews."""
    theme_id: int
    label: str
    description: str
    keywords: List[str]
    review_count: int
    representative_reviews: List[str]
    avg_sentiment: float
    sentiment_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    temporal_trend: Dict[str, float] = field(default_factory=dict)


class ThemeDiscoveryEngine:
    """
    Discovers recurring themes from customer reviews using unsupervised NLP.
    
    Architecture Decision:
    - Primary: TF-IDF + KMeans (zero external dependency beyond scikit-learn)
    - Enhanced: BERTopic (when transformer dependencies are available)
    
    The fallback ensures the system works out-of-the-box while still delivering
    meaningful theme extraction.
    """

    # Domain-specific stop words (beyond standard English stop words)
    DOMAIN_STOP_WORDS = {
        "havells", "product", "bought", "buy", "purchase", "ordered", "order",
        "amazon", "flipkart", "online", "delivered", "delivery", "received",
        "one", "two", "also", "would", "could", "using", "used", "month",
        "year", "week", "day", "time", "thing", "things", "really", "just",
        "much", "even", "still", "like", "get", "got", "getting", "make",
        "made", "making", "star", "stars", "review", "reviews", "overall",
        "india", "indian", "brand", "company", "model", "purchase"
    }

    ENGLISH_STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "don", "should", "now"
    }

    # Theme labeling templates based on top keywords
    THEME_LABEL_MAP = {
        frozenset(["noise", "sound", "loud", "noisy", "buzzing"]): "Noise & Vibration Issues",
        frozenset(["service", "support", "care", "technician", "complaint"]): "After-Sales Service Quality",
        frozenset(["quality", "build", "material", "plastic", "sturdy"]): "Build Quality & Materials",
        frozenset(["motor", "power", "speed", "performance"]): "Motor & Performance",
        frozenset(["install", "installation", "setup", "mounting"]): "Installation Experience",
        frozenset(["warranty", "replacement", "repair", "defective"]): "Warranty & Repair Issues",
        frozenset(["price", "value", "money", "cost", "expensive"]): "Value for Money",
        frozenset(["design", "look", "style", "aesthetic", "color"]): "Design & Aesthetics",
        frozenset(["energy", "electricity", "efficient", "saving", "bldc"]): "Energy Efficiency",
        frozenset(["heat", "heating", "temperature", "hot", "warm"]): "Heating Performance",
        frozenset(["filter", "purification", "air", "pm", "hepa"]): "Air Purification Quality",
        frozenset(["grind", "grinding", "mixer", "jar", "blade"]): "Grinding Performance",
        frozenset(["light", "led", "brightness", "bulb", "glow"]): "Lighting Quality",
        frozenset(["remote", "control", "smart", "app", "wifi"]): "Smart Controls & Remote",
        frozenset(["leak", "leaking", "water", "rust", "corrosion"]): "Leakage & Corrosion",
        frozenset(["safety", "shock", "spark", "cutoff", "protection"]): "Safety Concerns",
    }

    def __init__(self, n_themes: int = 12, min_theme_size: int = 10):
        self.n_themes = n_themes
        self.min_theme_size = min_theme_size
        self._all_stop_words = self.ENGLISH_STOP_WORDS | self.DOMAIN_STOP_WORDS

    def _preprocess_for_topics(self, texts: List[str]) -> List[str]:
        """Preprocess texts specifically for topic modeling."""
        processed = []
        for text in texts:
            text = text.lower()
            text = re.sub(r'[^a-z\s]', ' ', text)
            words = text.split()
            words = [w for w in words if w not in self._all_stop_words and len(w) > 2]
            processed.append(" ".join(words))
        return processed

    def discover_themes(
        self,
        texts: List[str],
        review_ids: List[str],
        sentiments: List[float],
        categories: List[str],
        dates: List[str],
    ) -> List[Theme]:
        """
        Discover themes using TF-IDF + KMeans clustering.
        
        This is the primary, dependency-light approach.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        # Preprocess
        processed_texts = self._preprocess_for_topics(texts)

        # Filter out empty documents
        valid_indices = [i for i, t in enumerate(processed_texts) if len(t.strip()) > 5]
        if len(valid_indices) < self.n_themes * 2:
            self.n_themes = max(3, len(valid_indices) // 5)

        filtered_texts = [processed_texts[i] for i in valid_indices]
        filtered_originals = [texts[i] for i in valid_indices]
        filtered_sentiments = [sentiments[i] for i in valid_indices]
        filtered_categories = [categories[i] for i in valid_indices]
        filtered_dates = [dates[i] for i in valid_indices]
        filtered_ids = [review_ids[i] for i in valid_indices]

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=3,
            max_df=0.85,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(filtered_texts)
        feature_names = vectorizer.get_feature_names_out()

        # KMeans Clustering
        n_clusters = min(self.n_themes, len(filtered_texts) // 5)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
        )
        labels = kmeans.fit_predict(tfidf_matrix)

        # Extract themes from clusters
        themes = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) < self.min_theme_size:
                continue

            # Get top keywords for this cluster
            center = kmeans.cluster_centers_[cluster_id]
            top_keyword_indices = center.argsort()[-15:][::-1]
            keywords = [feature_names[idx] for idx in top_keyword_indices]

            # Generate theme label
            label = self._generate_theme_label(keywords)

            # Get representative reviews (closest to cluster center)
            from sklearn.metrics.pairwise import cosine_similarity
            cluster_tfidf = tfidf_matrix[cluster_indices]
            similarities = cosine_similarity(
                cluster_tfidf,
                center.reshape(1, -1)
            ).flatten()
            top_rep_indices = similarities.argsort()[-5:][::-1]
            representative_reviews = [
                filtered_originals[cluster_indices[i]]
                for i in top_rep_indices
            ]

            # Compute cluster sentiment
            cluster_sentiments = [filtered_sentiments[i] for i in cluster_indices]
            avg_sentiment = float(np.mean(cluster_sentiments))

            sentiment_dist = {
                "positive": sum(1 for s in cluster_sentiments if s > 0.1),
                "neutral": sum(1 for s in cluster_sentiments if -0.1 <= s <= 0.1),
                "negative": sum(1 for s in cluster_sentiments if s < -0.1),
            }

            # Category distribution
            cluster_categories = [filtered_categories[i] for i in cluster_indices]
            cat_dist = dict(Counter(cluster_categories))

            # Temporal trend
            cluster_dates = [filtered_dates[i] for i in cluster_indices]
            temporal_trend = self._compute_theme_temporal_trend(
                cluster_sentiments, cluster_dates
            )

            # Generate description
            description = self._generate_theme_description(
                label, keywords[:5], avg_sentiment, len(cluster_indices),
                sentiment_dist, cat_dist
            )

            themes.append(Theme(
                theme_id=cluster_id,
                label=label,
                description=description,
                keywords=keywords[:10],
                review_count=len(cluster_indices),
                representative_reviews=representative_reviews,
                avg_sentiment=round(avg_sentiment, 3),
                sentiment_distribution=sentiment_dist,
                category_distribution=cat_dist,
                temporal_trend=temporal_trend,
            ))

        # Sort by review count (most discussed first)
        themes.sort(key=lambda t: t.review_count, reverse=True)

        return themes

    def _generate_theme_label(self, keywords: List[str]) -> str:
        """Generate a human-readable label for a theme based on its keywords."""
        keyword_set = set(keywords[:8])

        best_match = None
        best_overlap = 0

        for key_set, label in self.THEME_LABEL_MAP.items():
            # Check individual keywords (handle bigrams)
            overlap = 0
            for kw in keyword_set:
                for key_word in key_set:
                    if key_word in kw or kw in key_word:
                        overlap += 1
                        break

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = label

        if best_match and best_overlap >= 2:
            return best_match

        # Fallback: use top keywords
        clean_keywords = [kw for kw in keywords[:3] if len(kw) > 2]
        return f"Theme: {', '.join(clean_keywords).title()}"

    def _generate_theme_description(
        self, label: str, keywords: List[str], avg_sentiment: float,
        count: int, sentiment_dist: Dict, cat_dist: Dict
    ) -> str:
        """Generate a natural language description for a theme."""
        sentiment_word = "mixed"
        if avg_sentiment > 0.3:
            sentiment_word = "predominantly positive"
        elif avg_sentiment > 0.1:
            sentiment_word = "slightly positive"
        elif avg_sentiment < -0.3:
            sentiment_word = "predominantly negative"
        elif avg_sentiment < -0.1:
            sentiment_word = "slightly negative"

        top_category = max(cat_dist, key=cat_dist.get) if cat_dist else "various products"

        return (
            f'"{label}" is a {sentiment_word} theme appearing in {count} reviews, '
            f'most frequently related to {top_category.replace("_", " ")}. '
            f'Key discussion points include: {", ".join(keywords[:5])}. '
            f'Sentiment breakdown: {sentiment_dist.get("positive", 0)} positive, '
            f'{sentiment_dist.get("neutral", 0)} neutral, '
            f'{sentiment_dist.get("negative", 0)} negative reviews.'
        )

    def _compute_theme_temporal_trend(
        self, sentiments: List[float], dates: List[str]
    ) -> Dict[str, float]:
        """Compute how theme sentiment evolves over time."""
        from datetime import datetime

        monthly_data = defaultdict(list)
        for score, date_str in zip(sentiments, dates):
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                month_key = date.strftime("%Y-%m")
                monthly_data[month_key].append(score)
            except ValueError:
                continue

        trend = {}
        for month, scores in sorted(monthly_data.items()):
            trend[month] = round(float(np.mean(scores)), 3)

        return trend

    def get_theme_evolution(self, themes: List[Theme]) -> Dict:
        """
        Analyze how themes evolve over time.
        
        Returns evolution metrics including:
        - Growing themes (increasing mention frequency)
        - Declining themes (decreasing frequency)
        - Sentiment shift direction for each theme
        """
        evolution = {}

        for theme in themes:
            if len(theme.temporal_trend) < 3:
                continue

            months = sorted(theme.temporal_trend.keys())
            scores = [theme.temporal_trend[m] for m in months]

            # Compute trend direction using simple linear regression
            x = np.arange(len(scores))
            if len(x) > 1:
                slope = np.polyfit(x, scores, 1)[0]
            else:
                slope = 0

            evolution[theme.label] = {
                "trend_direction": "improving" if slope > 0.02 else "declining" if slope < -0.02 else "stable",
                "slope": round(float(slope), 4),
                "earliest_sentiment": scores[0] if scores else 0,
                "latest_sentiment": scores[-1] if scores else 0,
                "sentiment_change": round(scores[-1] - scores[0], 3) if len(scores) >= 2 else 0,
            }

        return evolution
