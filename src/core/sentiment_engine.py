"""
Sentiment Analysis Engine with Aspect-Based Sentiment Analysis (ABSA).

Provides multi-granularity sentiment analysis:
1. Document-level: Overall sentiment of a review
2. Aspect-level: Sentiment per product aspect (ABSA)
3. Temporal: Sentiment trends over time
4. Comparative: Cross-product and cross-category analysis
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

import numpy as np


@dataclass
class AspectSentiment:
    """Sentiment for a specific product aspect."""
    aspect: str
    sentiment_score: float  # -1.0 to 1.0
    sentiment_label: str    # very_negative, negative, neutral, positive, very_positive
    confidence: float       # 0.0 to 1.0
    evidence_snippets: List[str] = field(default_factory=list)
    mention_count: int = 0


@dataclass
class ReviewSentiment:
    """Complete sentiment analysis result for a single review."""
    review_id: str
    overall_score: float
    overall_label: str
    aspect_sentiments: List[AspectSentiment] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    emotion_indicators: Dict[str, float] = field(default_factory=dict)


class SentimentAnalyzer:
    """
    Multi-level sentiment analyzer combining lexicon-based and pattern-based approaches.
    
    Design Decision: We use a hybrid approach rather than relying solely on an LLM for 
    sentiment analysis because:
    1. Deterministic and reproducible results for the same input
    2. Much faster processing for batch analysis (no API calls)
    3. Transparent scoring that can be audited
    4. LLM is used as a secondary layer for nuanced/ambiguous cases
    
    The LLM is reserved for the QA agent where natural language understanding is critical.
    """

    # Enhanced sentiment lexicon with domain-specific terms
    POSITIVE_LEXICON = {
        # General positive
        "excellent": 0.9, "amazing": 0.85, "fantastic": 0.85, "superb": 0.9,
        "outstanding": 0.9, "perfect": 0.95, "wonderful": 0.85, "love": 0.8,
        "great": 0.7, "good": 0.5, "nice": 0.4, "decent": 0.3, "fine": 0.2,
        "happy": 0.7, "satisfied": 0.6, "impressed": 0.75, "recommend": 0.7,
        "best": 0.85, "premium": 0.6, "worth": 0.5, "reliable": 0.65,
        "efficient": 0.6, "smooth": 0.55, "quiet": 0.5, "sturdy": 0.6,
        "durable": 0.6, "powerful": 0.55, "stylish": 0.5, "elegant": 0.55,
        "comfortable": 0.5, "convenient": 0.45, "innovative": 0.6,
        # Domain-specific positive
        "energy-saving": 0.6, "silent": 0.55, "sleek": 0.5, "quick-heating": 0.6,
        "safe": 0.5, "value-for-money": 0.65, "fast": 0.45, "bright": 0.45,
    }

    NEGATIVE_LEXICON = {
        # General negative
        "terrible": -0.9, "horrible": -0.9, "worst": -0.95, "awful": -0.85,
        "pathetic": -0.9, "poor": -0.6, "bad": -0.5, "disappointing": -0.7,
        "disappointed": -0.7, "useless": -0.8, "waste": -0.75, "defective": -0.8,
        "broken": -0.7, "faulty": -0.75, "regret": -0.7, "avoid": -0.75,
        "cheap": -0.4, "frustrating": -0.7, "frustrated": -0.7, "unacceptable": -0.8,
        "mediocre": -0.3, "substandard": -0.65, "unreliable": -0.7,
        # Domain-specific negative
        "wobbling": -0.6, "leaking": -0.7, "overheating": -0.65, "flickering": -0.6,
        "noisy": -0.5, "rusted": -0.65, "cracked": -0.7, "burned": -0.8,
        "malfunctioning": -0.75, "unresponsive": -0.6,
    }

    # Negation words that flip sentiment
    NEGATION_WORDS = {
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "nor", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
        "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
        "hadn't", "shouldn't", "wouldn't", "couldn't", "mustn't"
    }

    # Intensifiers and diminishers
    INTENSIFIERS = {
        "very": 1.3, "really": 1.25, "extremely": 1.5, "incredibly": 1.4,
        "absolutely": 1.4, "totally": 1.3, "completely": 1.3, "highly": 1.3,
        "super": 1.3, "truly": 1.2, "exceptionally": 1.4, "remarkably": 1.3,
    }

    DIMINISHERS = {
        "slightly": 0.7, "somewhat": 0.75, "fairly": 0.8, "rather": 0.85,
        "a bit": 0.7, "a little": 0.7, "kind of": 0.75, "sort of": 0.75,
    }

    # Aspect extraction patterns with sentiment context
    ASPECT_PATTERNS = {
        "performance": {
            "keywords": ["performance", "speed", "power", "output", "air delivery",
                         "heating", "grinding", "purification", "brightness"],
            "positive_context": ["good", "great", "excellent", "fast", "powerful", "strong"],
            "negative_context": ["slow", "weak", "poor", "low", "inadequate", "drops"],
        },
        "build_quality": {
            "keywords": ["quality", "build", "material", "plastic", "metal", "construction",
                         "finish", "body", "casing"],
            "positive_context": ["premium", "sturdy", "solid", "durable", "robust"],
            "negative_context": ["cheap", "flimsy", "thin", "fragile", "poor"],
        },
        "noise": {
            "keywords": ["noise", "noisy", "sound", "buzzing", "humming", "clicking", "vibration", "loud", "quiet"],
            "positive_context": ["silent", "quiet", "noiseless", "smooth"],
            "negative_context": ["noisy", "loud", "rattling", "buzzing", "irritating"],
        },
        "after_sales_service": {
            "keywords": ["service", "support", "customer care", "technician", "repair",
                         "warranty", "complaint", "helpline", "service center"],
            "positive_context": ["helpful", "responsive", "quick", "professional", "resolved"],
            "negative_context": ["useless", "unresponsive", "delayed", "rude", "ignored", "pathetic"],
        },
        "installation": {
            "keywords": ["install", "installation", "setup", "mounting", "fitting", "assembly"],
            "positive_context": ["easy", "smooth", "quick", "simple", "hassle-free"],
            "negative_context": ["difficult", "complicated", "messy", "delayed", "extra cost"],
        },
        "energy_efficiency": {
            "keywords": ["energy", "electricity", "bill", "power consumption", "watt",
                         "efficient", "BLDC", "star rating", "BEE"],
            "positive_context": ["efficient", "saving", "low", "economical"],
            "negative_context": ["high", "expensive", "consuming", "wasteful"],
        },
        "design_aesthetics": {
            "keywords": ["design", "look", "appearance", "style", "color", "aesthetic",
                         "decor", "finish", "compact", "slim"],
            "positive_context": ["beautiful", "elegant", "sleek", "modern", "stylish", "premium"],
            "negative_context": ["ugly", "bulky", "old-fashioned", "cheap-looking"],
        },
        "safety": {
            "keywords": ["safety", "safe", "shock", "leak", "overheat", "spark",
                         "auto-cutoff", "ISI", "protection"],
            "positive_context": ["safe", "protected", "secure", "certified"],
            "negative_context": ["dangerous", "risky", "hazard", "shock", "leaking", "spark"],
        },
        "value_for_money": {
            "keywords": ["price", "value", "cost", "money", "worth", "budget",
                         "expensive", "affordable", "economical"],
            "positive_context": ["worth", "value", "affordable", "reasonable", "bargain"],
            "negative_context": ["expensive", "overpriced", "not worth", "waste of money"],
        },
        "durability": {
            "keywords": ["durability", "lasting", "lifespan", "long-term", "years",
                         "months", "broke", "failed", "stopped"],
            "positive_context": ["durable", "lasting", "years", "reliable", "robust"],
            "negative_context": ["broke", "failed", "stopped", "died", "short-lived", "within months"],
        },
    }

    def analyze_document_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Compute document-level sentiment score using enhanced lexicon approach.
        
        Returns:
            Tuple of (score: float [-1,1], label: str)
        """
        words = text.lower().split()
        scores = []
        i = 0

        while i < len(words):
            word = re.sub(r'[^\w\'-]', '', words[i])

            # Check for negation in preceding words
            is_negated = False
            if i > 0:
                prev_word = re.sub(r'[^\w\'-]', '', words[i - 1])
                if prev_word in self.NEGATION_WORDS:
                    is_negated = True
            if i > 1:
                prev_two = re.sub(r'[^\w\'-]', '', words[i - 2])
                if prev_two in self.NEGATION_WORDS:
                    is_negated = True

            # Check for intensifier
            intensity = 1.0
            if i > 0:
                prev_word = re.sub(r'[^\w\'-]', '', words[i - 1])
                if prev_word in self.INTENSIFIERS:
                    intensity = self.INTENSIFIERS[prev_word]
                elif prev_word in self.DIMINISHERS:
                    intensity = self.DIMINISHERS[prev_word]

            # Look up sentiment
            score = None
            if word in self.POSITIVE_LEXICON:
                score = self.POSITIVE_LEXICON[word] * intensity
            elif word in self.NEGATIVE_LEXICON:
                score = self.NEGATIVE_LEXICON[word] * intensity

            if score is not None:
                if is_negated:
                    score *= -0.75  # Partial negation (not fully inverted)
                scores.append(score)

            i += 1

        # Compute aggregate
        if not scores:
            return 0.0, "neutral"

        # Weighted average favoring extreme sentiments
        weighted_scores = []
        for s in scores:
            weight = abs(s)  # Stronger sentiments get more weight
            weighted_scores.append(s * weight)

        avg_score = sum(weighted_scores) / sum(abs(s) for s in scores)
        avg_score = max(-1.0, min(1.0, avg_score))

        # Map to label
        label = self._score_to_label(avg_score)

        return round(avg_score, 3), label

    def analyze_aspect_sentiment(self, text: str) -> List[AspectSentiment]:
        """
        Perform Aspect-Based Sentiment Analysis (ABSA).
        
        Identifies product aspects mentioned in the text and determines
        sentiment for each aspect independently.
        """
        text_lower = text.lower()
        sentences = re.split(r'[.!?]+', text)
        aspect_results = []

        for aspect_name, patterns in self.ASPECT_PATTERNS.items():
            # Check if aspect is mentioned
            mentioned = False
            evidence = []

            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                if not sentence_lower:
                    continue

                for keyword in patterns["keywords"]:
                    if keyword.lower() in sentence_lower:
                        mentioned = True
                        evidence.append(sentence.strip())
                        break

            if not mentioned:
                continue

            # Determine aspect-specific sentiment
            pos_signals = sum(1 for ctx in patterns["positive_context"]
                              if ctx.lower() in text_lower)
            neg_signals = sum(1 for ctx in patterns["negative_context"]
                              if ctx.lower() in text_lower)

            # Also use the general lexicon on evidence sentences
            evidence_text = " ".join(evidence)
            lexicon_score, _ = self.analyze_document_sentiment(evidence_text)

            # Combine signals
            context_score = (pos_signals - neg_signals) / max(pos_signals + neg_signals, 1)
            combined_score = 0.6 * lexicon_score + 0.4 * context_score
            combined_score = max(-1.0, min(1.0, combined_score))

            confidence = min(1.0, (pos_signals + neg_signals + len(evidence)) / 5.0)

            aspect_results.append(AspectSentiment(
                aspect=aspect_name,
                sentiment_score=round(combined_score, 3),
                sentiment_label=self._score_to_label(combined_score),
                confidence=round(confidence, 2),
                evidence_snippets=evidence[:3],  # Keep top 3 evidence snippets
                mention_count=len(evidence),
            ))

        return aspect_results

    def analyze_review(self, review_id: str, text: str) -> ReviewSentiment:
        """Complete sentiment analysis for a single review."""
        overall_score, overall_label = self.analyze_document_sentiment(text)
        aspect_sentiments = self.analyze_aspect_sentiment(text)
        key_phrases = self._extract_key_phrases(text)
        emotions = self._detect_emotions(text)

        return ReviewSentiment(
            review_id=review_id,
            overall_score=overall_score,
            overall_label=overall_label,
            aspect_sentiments=aspect_sentiments,
            key_phrases=key_phrases,
            emotion_indicators=emotions,
        )

    def compute_temporal_trends(
        self,
        sentiments: List[ReviewSentiment],
        dates: List[str],
        categories: List[str],
        granularity: str = "monthly"
    ) -> Dict:
        """
        Compute sentiment trends over time, grouped by category.
        
        Returns:
            Dictionary with temporal trend data for visualization.
        """
        from collections import defaultdict

        # Group by period and category
        period_data = defaultdict(lambda: defaultdict(list))

        for sentiment, date_str, category in zip(sentiments, dates, categories):
            date = datetime.strptime(date_str, "%Y-%m-%d")

            if granularity == "monthly":
                period_key = date.strftime("%Y-%m")
            elif granularity == "quarterly":
                quarter = (date.month - 1) // 3 + 1
                period_key = f"{date.year}-Q{quarter}"
            else:
                period_key = date.strftime("%Y-%W")  # weekly

            period_data[period_key][category].append(sentiment.overall_score)

        # Compute aggregates
        trends = {}
        for period, cat_scores in sorted(period_data.items()):
            trends[period] = {}
            for category, scores in cat_scores.items():
                trends[period][category] = {
                    "mean_sentiment": round(np.mean(scores), 3),
                    "median_sentiment": round(float(np.median(scores)), 3),
                    "std_sentiment": round(float(np.std(scores)), 3),
                    "review_count": len(scores),
                    "negative_pct": round(sum(1 for s in scores if s < -0.1) / len(scores) * 100, 1),
                    "positive_pct": round(sum(1 for s in scores if s > 0.1) / len(scores) * 100, 1),
                }

        return trends

    def compute_aspect_summary(
        self,
        all_sentiments: List[ReviewSentiment],
        category_filter: Optional[str] = None
    ) -> Dict:
        """
        Aggregate aspect-level sentiment across all reviews.
        
        Returns aspect-level summary with scores, trends, and evidence.
        """
        aspect_data = defaultdict(lambda: {
            "scores": [], "labels": [], "evidence": [], "mention_count": 0
        })

        for sentiment in all_sentiments:
            for asp in sentiment.aspect_sentiments:
                data = aspect_data[asp.aspect]
                data["scores"].append(asp.sentiment_score)
                data["labels"].append(asp.sentiment_label)
                data["evidence"].extend(asp.evidence_snippets)
                data["mention_count"] += asp.mention_count

        summary = {}
        for aspect, data in aspect_data.items():
            scores = data["scores"]
            summary[aspect] = {
                "mean_score": round(np.mean(scores), 3),
                "median_score": round(float(np.median(scores)), 3),
                "mention_count": data["mention_count"],
                "total_reviews_mentioning": len(scores),
                "sentiment_distribution": {
                    "positive": sum(1 for l in data["labels"] if l in ("positive", "very_positive")),
                    "neutral": sum(1 for l in data["labels"] if l == "neutral"),
                    "negative": sum(1 for l in data["labels"] if l in ("negative", "very_negative")),
                },
                "top_evidence": list(set(data["evidence"]))[:5],
            }

        return dict(sorted(summary.items(), key=lambda x: x[1]["mean_score"]))

    def _score_to_label(self, score: float) -> str:
        """Map numeric score to sentiment label."""
        if score <= -0.6:
            return "very_negative"
        elif score <= -0.1:
            return "negative"
        elif score <= 0.1:
            return "neutral"
        elif score <= 0.6:
            return "positive"
        else:
            return "very_positive"

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key descriptive phrases from text."""
        phrases = []
        # Match adjective + noun patterns
        patterns = [
            r'\b(excellent|good|bad|poor|great|terrible|amazing|worst|best)\s+\w+\b',
            r'\b\w+\s+(quality|performance|service|value|design)\b',
            r'\b(not|never|no)\s+\w+\b',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if isinstance(matches, list) and matches:
                for m in matches:
                    if isinstance(m, tuple):
                        phrases.append(" ".join(m))
                    else:
                        phrases.append(m)

        return phrases[:5]

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotional indicators in text."""
        text_lower = text.lower()

        emotion_keywords = {
            "frustration": ["frustrated", "frustrating", "annoying", "irritating", "annoyed"],
            "satisfaction": ["satisfied", "happy", "pleased", "content", "delighted"],
            "anger": ["angry", "furious", "outraged", "disgusted", "livid"],
            "trust": ["reliable", "trustworthy", "dependable", "consistent"],
            "disappointment": ["disappointed", "let down", "expected more", "not up to"],
        }

        emotions = {}
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                emotions[emotion] = min(1.0, count / 2.0)

        return emotions
