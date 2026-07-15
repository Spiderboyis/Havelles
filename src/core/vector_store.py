"""
Vector Store Manager for Review Retrieval.

Implements a lightweight in-memory vector store using TF-IDF embeddings
and cosine similarity for grounded retrieval. This avoids external
dependencies (ChromaDB, FAISS) while maintaining retrieval quality.

For production, this can be swapped with ChromaDB or Pinecone.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    """A single retrieval result with relevance metadata."""
    review_id: str
    review_text: str
    product_name: str
    product_category: str
    rating: int
    review_date: str
    similarity_score: float
    relevant_snippet: str = ""


class VectorStoreManager:
    """
    Manages review embeddings and retrieval for grounded QA.
    
    Design Decision: We use TF-IDF vectors instead of dense embeddings
    (sentence-transformers) for the base implementation because:
    1. No GPU or large model download required
    2. Fast to build and query
    3. Excellent for keyword-heavy retrieval (review text is keyword-rich)
    4. Deterministic results
    
    For production with larger datasets (100K+ reviews), upgrade to
    sentence-transformers embeddings + FAISS or ChromaDB.
    """

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
        )
        self.tfidf_matrix = None
        self.documents = []  # Store full review data
        self._is_fitted = False

    def index_reviews(self, reviews: List[Dict]) -> int:
        """
        Build the vector index from a list of reviews.
        
        Args:
            reviews: List of review dictionaries with at least 'review_text' field
            
        Returns:
            Number of successfully indexed reviews
        """
        texts = []
        valid_reviews = []

        for review in reviews:
            text = review.get("cleaned_text", review.get("review_text", ""))
            if text and len(text.strip()) > 10:
                # Enrich text with metadata for better retrieval
                enriched = self._enrich_text(text, review)
                texts.append(enriched)
                valid_reviews.append(review)

        if not texts:
            return 0

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.documents = valid_reviews
        self._is_fitted = True

        return len(valid_reviews)

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        category_filter: Optional[str] = None,
        min_rating: Optional[int] = None,
        max_rating: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the most relevant reviews for a given query.
        
        Args:
            query: Natural language query from the product manager
            top_k: Number of results to return
            category_filter: Filter by product category
            min_rating/max_rating: Filter by rating range
            date_from/date_to: Filter by date range
            
        Returns:
            List of RetrievalResult objects sorted by relevance
        """
        if not self._is_fitted:
            raise RuntimeError("Vector store not initialized. Call index_reviews() first.")

        # Transform query
        query_vec = self.vectorizer.transform([query.lower()])

        # Compute similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply filters and get top results
        results = []
        for idx in similarities.argsort()[::-1]:
            if len(results) >= top_k * 3:  # Get extra for filtering
                break

            doc = self.documents[idx]
            score = float(similarities[idx])

            if score < 0.01:  # Minimum relevance threshold
                continue

            # Apply filters
            if category_filter and doc.get("product_category") != category_filter:
                continue

            rating = int(doc.get("rating", 0))
            if min_rating and rating < min_rating:
                continue
            if max_rating and rating > max_rating:
                continue

            if date_from and doc.get("review_date", "") < date_from:
                continue
            if date_to and doc.get("review_date", "") > date_to:
                continue

            # Extract relevant snippet
            review_text = doc.get("cleaned_text", doc.get("review_text", ""))
            snippet = self._extract_relevant_snippet(review_text, query)

            results.append(RetrievalResult(
                review_id=doc.get("review_id", ""),
                review_text=review_text,
                product_name=doc.get("product_name", ""),
                product_category=doc.get("product_category", ""),
                rating=rating,
                review_date=doc.get("review_date", ""),
                similarity_score=round(score, 4),
                relevant_snippet=snippet,
            ))

        # Return top_k results
        return results[:top_k]

    def retrieve_by_aspect(
        self, aspect: str, sentiment: Optional[str] = None, top_k: int = 10
    ) -> List[RetrievalResult]:
        """Retrieve reviews discussing a specific aspect with optional sentiment filter."""
        query = f"{aspect} quality performance"

        results = self.retrieve(query, top_k=top_k * 2)

        if sentiment:
            # Simple sentiment-based post-filtering
            filtered = []
            sentiment_keywords = {
                "positive": ["good", "great", "excellent", "happy", "love", "best", "perfect"],
                "negative": ["bad", "poor", "terrible", "worst", "broken", "defective", "disappointed"],
            }
            keywords = sentiment_keywords.get(sentiment, [])

            for result in results:
                text_lower = result.review_text.lower()
                if any(kw in text_lower for kw in keywords):
                    filtered.append(result)

            results = filtered if filtered else results

        return results[:top_k]

    def _enrich_text(self, text: str, review: Dict) -> str:
        """Enrich review text with metadata for better retrieval."""
        parts = [text.lower()]

        product = review.get("product_name", "")
        if product:
            parts.append(product.lower())

        category = review.get("product_category", "")
        if category:
            parts.append(category.replace("_", " ").lower())

        rating = review.get("rating", 0)
        if rating:
            if int(rating) <= 2:
                parts.append("negative complaint issue problem")
            elif int(rating) >= 4:
                parts.append("positive satisfied good quality")

        return " ".join(parts)

    def _extract_relevant_snippet(self, text: str, query: str, max_len: int = 200) -> str:
        """Extract the most relevant snippet from a review for the given query."""
        if len(text) <= max_len:
            return text

        query_words = set(query.lower().split())
        sentences = re.split(r'[.!?]+', text)

        if not sentences:
            return text[:max_len]

        # Score each sentence by query word overlap
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            scored.append((overlap, sent))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Build snippet from most relevant sentences
        snippet_parts = []
        current_len = 0
        for _, sent in scored:
            if current_len + len(sent) > max_len:
                break
            snippet_parts.append(sent)
            current_len += len(sent)

        return ". ".join(snippet_parts) if snippet_parts else text[:max_len]

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        if not self._is_fitted:
            return {"status": "not_initialized"}

        return {
            "total_documents": len(self.documents),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "matrix_shape": self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
            "categories": list(set(d.get("product_category", "") for d in self.documents)),
            "products": list(set(d.get("product_name", "") for d in self.documents)),
        }
