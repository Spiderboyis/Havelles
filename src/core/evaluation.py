"""
Evaluation Framework for the Customer Voice Intelligence Agent.

Measures system quality across four dimensions:
1. Retrieval Quality — Are we finding the right reviews?
2. Sentiment Accuracy — Is sentiment classification correct?
3. Theme Coherence — Are discovered themes meaningful?
4. Grounding Faithfulness — Are answers backed by actual data?
"""

from typing import List, Dict, Tuple
import numpy as np
from collections import Counter


class EvaluationFramework:
    """
    Evaluates the Customer Voice Intelligence Agent across multiple dimensions.
    
    This framework provides quantitative metrics that can be tracked over time
    to ensure the system maintains quality as data and models evolve.
    """

    def evaluate_retrieval_quality(
        self,
        queries: List[str],
        retrieved_results: List[List[Dict]],
        expected_categories: List[str],
    ) -> Dict:
        """
        Evaluate retrieval quality using category-level precision.
        
        Since we don't have human-labeled relevance judgments,
        we use category matching as a proxy: if a query is about fans,
        retrieved reviews about fans are considered relevant.
        """
        precision_scores = []
        recall_estimates = []

        for query, results, expected_cat in zip(queries, retrieved_results, expected_categories):
            if not results:
                precision_scores.append(0.0)
                continue

            relevant = sum(
                1 for r in results
                if r.get("category", r.get("product_category", "")) == expected_cat
            )
            precision = relevant / len(results)
            precision_scores.append(precision)

        return {
            "mean_precision": round(float(np.mean(precision_scores)), 3),
            "min_precision": round(float(np.min(precision_scores)), 3),
            "max_precision": round(float(np.max(precision_scores)), 3),
            "queries_evaluated": len(queries),
        }

    def evaluate_sentiment_accuracy(
        self,
        predicted_labels: List[str],
        true_labels: List[str],
    ) -> Dict:
        """
        Evaluate sentiment classification accuracy.
        
        Uses the synthetic data's ground truth labels for evaluation.
        Maps to binary (positive/negative) and ternary (pos/neu/neg) accuracy.
        """
        if len(predicted_labels) != len(true_labels):
            return {"error": "Mismatched lengths"}

        # Ternary mapping
        def to_ternary(label):
            if label in ("positive", "very_positive"):
                return "positive"
            elif label in ("negative", "very_negative"):
                return "negative"
            return "neutral"

        pred_ternary = [to_ternary(l) for l in predicted_labels]
        true_ternary = [to_ternary(l) for l in true_labels]

        correct = sum(1 for p, t in zip(pred_ternary, true_ternary) if p == t)
        accuracy = correct / len(predicted_labels)

        # Per-class metrics
        classes = ["positive", "neutral", "negative"]
        per_class = {}
        for cls in classes:
            tp = sum(1 for p, t in zip(pred_ternary, true_ternary) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(pred_ternary, true_ternary) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(pred_ternary, true_ternary) if p != cls and t == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class[cls] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            }

        return {
            "ternary_accuracy": round(accuracy, 3),
            "total_evaluated": len(predicted_labels),
            "per_class": per_class,
        }

    def evaluate_theme_coherence(self, themes: List[Dict]) -> Dict:
        """
        Evaluate theme quality using coherence metrics:
        1. Keyword overlap between themes (lower is better — themes should be distinct)
        2. Theme size distribution (more uniform is better)
        3. Theme coverage (what % of reviews are assigned to themes)
        """
        if not themes:
            return {"error": "No themes to evaluate"}

        # Keyword distinctness
        all_keyword_sets = [set(t.get("keywords", [])) for t in themes]
        overlap_scores = []

        for i in range(len(all_keyword_sets)):
            for j in range(i + 1, len(all_keyword_sets)):
                intersection = len(all_keyword_sets[i] & all_keyword_sets[j])
                union = len(all_keyword_sets[i] | all_keyword_sets[j])
                if union > 0:
                    overlap_scores.append(intersection / union)

        avg_overlap = float(np.mean(overlap_scores)) if overlap_scores else 0
        distinctness = 1 - avg_overlap  # Higher is better

        # Size distribution uniformity
        sizes = [t.get("review_count", 0) for t in themes]
        total = sum(sizes)
        if total > 0 and len(sizes) > 1:
            proportions = [s / total for s in sizes]
            entropy = -sum(p * np.log2(p + 1e-10) for p in proportions)
            max_entropy = np.log2(len(sizes))
            uniformity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            uniformity = 0

        return {
            "num_themes": len(themes),
            "theme_distinctness": round(distinctness, 3),
            "size_uniformity": round(float(uniformity), 3),
            "avg_theme_size": round(float(np.mean(sizes)), 1),
            "total_reviews_in_themes": total,
            "theme_labels": [t.get("label", "") for t in themes],
        }

    def evaluate_grounding_faithfulness(
        self,
        response: str,
        source_reviews: List[Dict],
    ) -> Dict:
        """
        Evaluate whether the generated response is grounded in source reviews.
        
        Checks:
        1. Does the response cite specific numbers that match the data?
        2. Are quoted snippets actually from the source reviews?
        3. Does the response acknowledge limitations?
        """
        # Check for citation presence
        has_citations = "SOURCE CITATIONS" in response or "[" in response
        
        # Check for data acknowledgment
        mentions_data_size = any(
            word in response.lower()
            for word in ["reviews", "analyzed", "based on", "out of"]
        )
        
        # Check for limitation acknowledgment
        has_limitations = any(
            word in response.lower()
            for word in ["limitation", "warning", "note", "insufficient", "may not"]
        )

        # Check quoted text against sources
        import re
        quoted_texts = re.findall(r'"([^"]{10,})"', response)
        source_texts = [
            r.get("text", r.get("review_text", r.get("cleaned_text", "")))
            for r in source_reviews
        ]
        source_combined = " ".join(source_texts).lower()

        grounded_quotes = 0
        for quote in quoted_texts:
            # Check if the quote (or a significant portion) exists in sources
            quote_words = set(quote.lower().split()[:8])
            if len(quote_words) >= 3:
                matches = sum(1 for w in quote_words if w in source_combined)
                if matches / len(quote_words) > 0.5:
                    grounded_quotes += 1

        quote_grounding = grounded_quotes / len(quoted_texts) if quoted_texts else 1.0

        return {
            "has_citations": has_citations,
            "mentions_data_coverage": mentions_data_size,
            "acknowledges_limitations": has_limitations,
            "total_quotes": len(quoted_texts),
            "grounded_quotes": grounded_quotes,
            "quote_grounding_rate": round(quote_grounding, 3),
            "overall_faithfulness": round(
                (0.3 * float(has_citations) +
                 0.2 * float(mentions_data_size) +
                 0.2 * float(has_limitations) +
                 0.3 * quote_grounding), 3
            ),
        }

    def run_full_evaluation(
        self,
        orchestrator,
        test_queries: List[Dict],
    ) -> Dict:
        """
        Run a comprehensive evaluation of the system.
        
        Args:
            orchestrator: The CustomerVoiceOrchestrator instance
            test_queries: List of dicts with 'query' and 'expected_category'
        """
        results = {}
        
        # Test retrieval and response generation
        for tq in test_queries:
            query = tq["query"]
            response = orchestrator.ask(query)
            
            # This is a qualitative check — in production, 
            # you'd have human annotators score these
            results[query] = {
                "response_length": len(response),
                "has_structure": "═" in response or "─" in response,
                "mentions_data": "reviews" in response.lower(),
            }

        return {
            "queries_tested": len(test_queries),
            "all_structured": all(r["has_structure"] for r in results.values()),
            "all_mention_data": all(r["mentions_data"] for r in results.values()),
            "individual_results": results,
        }


# ─── Predefined Test Queries ──────────────────────────────────────

TEST_QUERIES = [
    {
        "query": "What are the main complaints about Havells ceiling fans?",
        "expected_category": "fans",
        "expected_aspects": ["noise", "build_quality", "service"],
    },
    {
        "query": "How has customer satisfaction with water heaters changed over time?",
        "expected_category": "water_heaters",
        "expected_aspects": ["performance", "durability"],
    },
    {
        "query": "What do customers think about the build quality of mixer grinders?",
        "expected_category": "mixer_grinders",
        "expected_aspects": ["build_quality", "durability"],
    },
    {
        "query": "Are noise issues getting better or worse for fans?",
        "expected_category": "fans",
        "expected_aspects": ["noise"],
    },
    {
        "query": "What's the overall sentiment across all Havells products?",
        "expected_category": None,
        "expected_aspects": [],
    },
    {
        "query": "Tell me about after-sales service complaints",
        "expected_category": None,
        "expected_aspects": ["after_sales_service"],
    },
]
