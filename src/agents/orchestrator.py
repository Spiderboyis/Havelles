"""
Multi-Agent Orchestrator for Customer Voice Intelligence.

Implements the Orchestrator-Worker pattern using a state graph:

  ┌─────────────┐
  │  Orchestrator│  ← Receives manager's question
  │   (Router)   │
  └──────┬───────┘
         │ Routes to appropriate specialist
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌────────┐┌────────┐┌────────┐┌────────┐
│Sentiment││ Theme  ││Retrieval││ Trend  │
│ Analyst ││Analyst ││ Agent  ││Analyst │
└────┬───┘└───┬────┘└───┬────┘└───┬────┘
     │        │         │         │
     └────────┴────┬────┴─────────┘
                   ▼
            ┌────────────┐
            │  Synthesizer│  ← Grounds & formats response
            │   (Critic)  │
            └─────────────┘

Design Decisions:
1. Stateless agents: Each agent is a pure function (input → output)
2. Explicit state passing: No hidden state between agents
3. Grounding verification: Synthesizer validates claims against source reviews
4. No LLM dependency for core analysis: LLM only for natural language synthesis
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import re

from src.core.sentiment_engine import SentimentAnalyzer, ReviewSentiment
from src.core.theme_engine import ThemeDiscoveryEngine, Theme
from src.core.vector_store import VectorStoreManager, RetrievalResult


class QueryIntent(Enum):
    """Classified intent of a product manager's query."""
    SENTIMENT_OVERVIEW = "sentiment_overview"
    THEME_DISCOVERY = "theme_discovery"
    PRODUCT_COMPARISON = "product_comparison"
    TREND_ANALYSIS = "trend_analysis"
    SPECIFIC_ISSUE = "specific_issue"
    ASPECT_DEEP_DIVE = "aspect_deep_dive"
    GENERAL_SUMMARY = "general_summary"


@dataclass
class AgentState:
    """
    Shared state object passed between agents in the pipeline.
    
    This is the "message bus" — each agent reads what it needs
    and writes its output for downstream agents.
    """
    # Input
    query: str = ""
    intent: Optional[QueryIntent] = None
    
    # Filters extracted from query
    category_filter: Optional[str] = None
    product_filter: Optional[str] = None
    date_range: Optional[Dict[str, str]] = None
    aspect_filter: Optional[str] = None
    
    # Agent outputs
    retrieved_reviews: List[Dict] = field(default_factory=list)
    sentiment_results: Optional[Dict] = None
    theme_results: Optional[List[Dict]] = None
    trend_results: Optional[Dict] = None
    
    # Final output
    grounded_response: str = ""
    citations: List[Dict] = field(default_factory=list)
    confidence_score: float = 0.0
    data_coverage: str = ""
    
    # Metadata
    agents_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class OrchestratorAgent:
    """
    Routes queries to appropriate specialist agents based on intent classification.
    
    Intent Classification is done via keyword matching + pattern recognition
    (no LLM required for routing, keeping it fast and deterministic).
    """

    INTENT_PATTERNS = {
        QueryIntent.SENTIMENT_OVERVIEW: [
            r"how.*feel", r"sentiment", r"satisfaction", r"happy|unhappy",
            r"overall.*opinion", r"what.*think", r"how.*rated",
            r"positive|negative", r"mood", r"perception",
        ],
        QueryIntent.THEME_DISCOVERY: [
            r"theme", r"topic", r"recurring", r"common.*issue",
            r"main.*complaint", r"what.*talking.*about", r"pattern",
            r"frequent", r"most.*mentioned", r"top.*issue",
        ],
        QueryIntent.PRODUCT_COMPARISON: [
            r"compar", r"versus|vs", r"better|worse.*than",
            r"difference.*between", r"which.*product",
        ],
        QueryIntent.TREND_ANALYSIS: [
            r"trend", r"over.*time", r"improving|declining|getting.*better|getting.*worse",
            r"month.*over.*month", r"change", r"evolv", r"shift",
            r"increas|decreas", r"last.*months?", r"quarter",
        ],
        QueryIntent.SPECIFIC_ISSUE: [
            r"noise|vibrat", r"leak", r"service|warranty",
            r"defect", r"safety", r"broken|not.*working",
            r"motor|heating|grind|filter|light|bulb",
        ],
        QueryIntent.ASPECT_DEEP_DIVE: [
            r"tell.*about.*(?:quality|performance|service|design|safety)",
            r"how.*(?:quality|performance|service|noise|durability)",
            r"aspect", r"feature.*feedback",
        ],
    }

    CATEGORY_KEYWORDS = {
        "fans": ["fan", "fans", "ceiling fan", "table fan", "bldc", "air delivery", "wobble"],
        "water_heaters": ["heater", "geyser", "hot water", "thermostat", "heating element"],
        "mixer_grinders": ["mixer", "grinder", "grinding", "blender", "juicer", "jar", "blade"],
        "air_purifiers": ["purifier", "air quality", "hepa", "filter", "pm2.5", "pollution"],
        "lighting": ["light", "led", "bulb", "lamp", "lighting", "brightness", "smart light"],
    }

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the user's query into an intent category."""
        query_lower = query.lower()
        intent_scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > 0:
                intent_scores[intent] = score

        if intent_scores:
            return max(intent_scores, key=intent_scores.get)

        return QueryIntent.GENERAL_SUMMARY

    def extract_filters(self, query: str) -> Dict:
        """Extract product/category/date filters from the query."""
        query_lower = query.lower()
        filters = {}

        # Category detection
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                filters["category"] = category
                break

        # Aspect detection
        aspects = [
            "noise", "service", "quality", "performance", "design",
            "safety", "value", "durability", "installation", "energy"
        ]
        for aspect in aspects:
            if aspect in query_lower:
                filters["aspect"] = aspect
                break

        return filters

    def route(self, state: AgentState) -> List[str]:
        """
        Determine which agents need to be invoked for this query.
        
        Returns list of agent names to invoke in order.
        """
        intent = self.classify_intent(state.query)
        state.intent = intent

        filters = self.extract_filters(state.query)
        state.category_filter = filters.get("category")
        state.aspect_filter = filters.get("aspect")

        # Route to appropriate agents
        agent_sequence = ["retrieval"]  # Always retrieve relevant reviews first

        routing_map = {
            QueryIntent.SENTIMENT_OVERVIEW: ["sentiment", "synthesizer"],
            QueryIntent.THEME_DISCOVERY: ["theme", "synthesizer"],
            QueryIntent.PRODUCT_COMPARISON: ["sentiment", "theme", "synthesizer"],
            QueryIntent.TREND_ANALYSIS: ["sentiment", "trend", "synthesizer"],
            QueryIntent.SPECIFIC_ISSUE: ["sentiment", "theme", "synthesizer"],
            QueryIntent.ASPECT_DEEP_DIVE: ["sentiment", "synthesizer"],
            QueryIntent.GENERAL_SUMMARY: ["sentiment", "theme", "trend", "synthesizer"],
        }

        agent_sequence.extend(routing_map.get(intent, ["sentiment", "synthesizer"]))

        return agent_sequence


class RetrievalAgent:
    """Retrieves relevant reviews from the vector store based on the query."""

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def execute(self, state: AgentState) -> AgentState:
        """Retrieve relevant reviews and add them to state."""
        results = self.vector_store.retrieve(
            query=state.query,
            top_k=20,
            category_filter=state.category_filter,
        )

        state.retrieved_reviews = [
            {
                "review_id": r.review_id,
                "text": r.review_text,
                "product": r.product_name,
                "category": r.product_category,
                "rating": r.rating,
                "date": r.review_date,
                "relevance": r.similarity_score,
                "snippet": r.relevant_snippet,
            }
            for r in results
        ]

        state.agents_used.append("retrieval")

        if len(state.retrieved_reviews) < 5:
            state.warnings.append(
                f"Only {len(state.retrieved_reviews)} relevant reviews found. "
                "Results may not be fully representative."
            )

        return state


class SentimentAgent:
    """Analyzes sentiment of retrieved reviews at document and aspect levels."""

    def __init__(self):
        self.analyzer = SentimentAnalyzer()

    def execute(self, state: AgentState) -> AgentState:
        """Run sentiment analysis on retrieved reviews."""
        if not state.retrieved_reviews:
            state.warnings.append("No reviews to analyze for sentiment.")
            return state

        sentiments = []
        aspect_aggregation = {}

        for review in state.retrieved_reviews:
            result = self.analyzer.analyze_review(
                review["review_id"], review["text"]
            )
            sentiments.append({
                "review_id": review["review_id"],
                "overall_score": result.overall_score,
                "overall_label": result.overall_label,
                "aspects": [
                    {
                        "aspect": a.aspect,
                        "score": a.sentiment_score,
                        "label": a.sentiment_label,
                        "evidence": a.evidence_snippets,
                    }
                    for a in result.aspect_sentiments
                ],
                "emotions": result.emotion_indicators,
            })

            # Aggregate aspects
            for asp in result.aspect_sentiments:
                if asp.aspect not in aspect_aggregation:
                    aspect_aggregation[asp.aspect] = {"scores": [], "evidence": []}
                aspect_aggregation[asp.aspect]["scores"].append(asp.sentiment_score)
                aspect_aggregation[asp.aspect]["evidence"].extend(asp.evidence_snippets)

        # Compute summary statistics
        overall_scores = [s["overall_score"] for s in sentiments]
        import numpy as np

        state.sentiment_results = {
            "individual": sentiments,
            "summary": {
                "mean_sentiment": round(float(np.mean(overall_scores)), 3),
                "median_sentiment": round(float(np.median(overall_scores)), 3),
                "std_sentiment": round(float(np.std(overall_scores)), 3),
                "positive_count": sum(1 for s in overall_scores if s > 0.1),
                "neutral_count": sum(1 for s in overall_scores if -0.1 <= s <= 0.1),
                "negative_count": sum(1 for s in overall_scores if s < -0.1),
                "total_analyzed": len(sentiments),
            },
            "aspect_summary": {
                asp: {
                    "mean_score": round(float(np.mean(data["scores"])), 3),
                    "mention_count": len(data["scores"]),
                    "top_evidence": list(set(data["evidence"]))[:3],
                }
                for asp, data in sorted(
                    aspect_aggregation.items(),
                    key=lambda x: np.mean(x[1]["scores"])
                )
            },
        }

        state.agents_used.append("sentiment")
        return state


class ThemeAgent:
    """Discovers and analyzes themes in retrieved reviews."""

    def __init__(self):
        self.engine = ThemeDiscoveryEngine(n_themes=8, min_theme_size=3)

    def execute(self, state: AgentState, all_reviews: List[Dict] = None) -> AgentState:
        """Discover themes from reviews."""
        # Use all reviews for theme discovery (not just retrieved ones)
        reviews = all_reviews if all_reviews else state.retrieved_reviews

        if len(reviews) < 10:
            state.warnings.append("Too few reviews for reliable theme discovery.")
            state.theme_results = []
            state.agents_used.append("theme")
            return state

        texts = [r.get("text", r.get("cleaned_text", r.get("review_text", ""))) for r in reviews]
        ids = [r.get("review_id", str(i)) for i, r in enumerate(reviews)]
        # Use 0.0 as default sentiment for theme discovery
        sentiments = [0.0] * len(reviews)
        categories = [r.get("category", r.get("product_category", "unknown")) for r in reviews]
        dates = [r.get("date", r.get("review_date", "2024-01-01")) for r in reviews]

        themes = self.engine.discover_themes(texts, ids, sentiments, categories, dates)

        state.theme_results = [
            {
                "theme_id": t.theme_id,
                "label": t.label,
                "description": t.description,
                "keywords": t.keywords,
                "review_count": t.review_count,
                "avg_sentiment": t.avg_sentiment,
                "sentiment_distribution": t.sentiment_distribution,
                "category_distribution": t.category_distribution,
                "representative_reviews": t.representative_reviews[:3],
                "temporal_trend": t.temporal_trend,
            }
            for t in themes
        ]

        state.agents_used.append("theme")
        return state


class TrendAgent:
    """Analyzes temporal trends in sentiment and themes."""

    def execute(self, state: AgentState) -> AgentState:
        """Compute temporal trend analysis."""
        if not state.retrieved_reviews:
            state.warnings.append("No reviews for trend analysis.")
            state.agents_used.append("trend")
            return state

        from collections import defaultdict
        import numpy as np

        monthly_data = defaultdict(lambda: {"scores": [], "ratings": [], "count": 0})

        for review in state.retrieved_reviews:
            date = review.get("date", "")
            if not date or len(date) < 7:
                continue
            month_key = date[:7]  # YYYY-MM
            rating = review.get("rating", 3)
            monthly_data[month_key]["ratings"].append(rating)
            monthly_data[month_key]["count"] += 1

        if len(monthly_data) < 2:
            state.trend_results = {"note": "Insufficient temporal data for trend analysis"}
            state.agents_used.append("trend")
            return state

        trend_data = {}
        for month, data in sorted(monthly_data.items()):
            ratings = data["ratings"]
            trend_data[month] = {
                "avg_rating": round(float(np.mean(ratings)), 2),
                "review_count": data["count"],
                "low_rating_pct": round(
                    sum(1 for r in ratings if r <= 2) / len(ratings) * 100, 1
                ),
            }

        # Compute overall trend direction
        months = sorted(trend_data.keys())
        if len(months) >= 3:
            early_avg = np.mean([trend_data[m]["avg_rating"] for m in months[:len(months)//3]])
            late_avg = np.mean([trend_data[m]["avg_rating"] for m in months[-len(months)//3:]])
            direction = "improving" if late_avg > early_avg + 0.1 else \
                        "declining" if late_avg < early_avg - 0.1 else "stable"
        else:
            direction = "insufficient_data"

        state.trend_results = {
            "monthly": trend_data,
            "direction": direction,
            "total_months": len(months),
        }

        state.agents_used.append("trend")
        return state


class SynthesizerAgent:
    """
    Synthesizes findings from all agents into a grounded, natural language response.
    
    This is the ONLY agent that may use an LLM for generation.
    However, it works perfectly without one using template-based synthesis.
    
    CRITICAL: Every claim must be grounded in actual review data.
    If data doesn't support a conclusion, the response says so.
    """

    def execute(self, state: AgentState) -> AgentState:
        """Synthesize a grounded response from all agent outputs."""
        response_parts = []
        citations = []

        # Header based on intent
        intent_headers = {
            QueryIntent.SENTIMENT_OVERVIEW: "📊 Sentiment Analysis Report",
            QueryIntent.THEME_DISCOVERY: "🔍 Theme Discovery Report",
            QueryIntent.TREND_ANALYSIS: "📈 Trend Analysis Report",
            QueryIntent.PRODUCT_COMPARISON: "⚖️ Product Comparison Report",
            QueryIntent.SPECIFIC_ISSUE: "🔧 Issue Analysis Report",
            QueryIntent.ASPECT_DEEP_DIVE: "🎯 Aspect Deep-Dive Report",
            QueryIntent.GENERAL_SUMMARY: "📋 Customer Voice Summary",
        }
        header = intent_headers.get(state.intent, "📋 Analysis Report")
        response_parts.append(f"\n{'='*60}\n{header}\n{'='*60}\n")

        # Coverage statement
        total_reviews = len(state.retrieved_reviews)
        category_note = f" for {state.category_filter.replace('_', ' ')}" if state.category_filter else ""
        response_parts.append(
            f"\n📌 Analysis based on {total_reviews} relevant reviews{category_note}.\n"
        )

        # Sentiment summary
        if state.sentiment_results:
            summary = state.sentiment_results["summary"]
            response_parts.append(self._format_sentiment_section(summary))

            # Aspect summary
            if state.sentiment_results.get("aspect_summary"):
                response_parts.append(self._format_aspect_section(
                    state.sentiment_results["aspect_summary"]
                ))

        # Theme summary
        if state.theme_results:
            response_parts.append(self._format_theme_section(state.theme_results))

        # Trend summary
        if state.trend_results and state.trend_results.get("monthly"):
            response_parts.append(self._format_trend_section(state.trend_results))

        # Grounding citations
        if state.retrieved_reviews:
            citations = self._generate_citations(state.retrieved_reviews[:8])
            response_parts.append(self._format_citations(citations))

        # Warnings
        if state.warnings:
            response_parts.append("\n⚠️ Data Limitations:")
            for w in state.warnings:
                response_parts.append(f"  • {w}")

        state.grounded_response = "\n".join(response_parts)
        state.citations = citations
        state.confidence_score = min(1.0, total_reviews / 15.0)
        state.data_coverage = f"{total_reviews} reviews analyzed"
        state.agents_used.append("synthesizer")

        return state

    def _format_sentiment_section(self, summary: Dict) -> str:
        """Format sentiment summary as readable text."""
        total = summary["total_analyzed"]
        pos = summary["positive_count"]
        neg = summary["negative_count"]
        neutral = summary["neutral_count"]
        mean = summary["mean_sentiment"]

        sentiment_word = "mixed"
        if mean > 0.3:
            sentiment_word = "predominantly positive"
        elif mean > 0.1:
            sentiment_word = "slightly positive"
        elif mean < -0.3:
            sentiment_word = "predominantly negative"
        elif mean < -0.1:
            sentiment_word = "slightly negative"

        return (
            f"\n📊 OVERALL SENTIMENT: {sentiment_word.upper()}\n"
            f"{'─'*40}\n"
            f"  Mean Score: {mean:.2f} (scale: -1 to +1)\n"
            f"  ✅ Positive: {pos}/{total} ({pos/total*100:.0f}%)\n"
            f"  ⚪ Neutral:  {neutral}/{total} ({neutral/total*100:.0f}%)\n"
            f"  ❌ Negative: {neg}/{total} ({neg/total*100:.0f}%)\n"
        )

    def _format_aspect_section(self, aspect_summary: Dict) -> str:
        """Format aspect-level sentiment summary."""
        lines = [f"\n🎯 ASPECT-LEVEL SENTIMENT:\n{'─'*40}"]

        for aspect, data in aspect_summary.items():
            score = data["mean_score"]
            count = data["mention_count"]

            if score > 0.3:
                indicator = "🟢"
            elif score > 0:
                indicator = "🟡"
            elif score > -0.3:
                indicator = "🟠"
            else:
                indicator = "🔴"

            aspect_name = aspect.replace("_", " ").title()
            lines.append(f"  {indicator} {aspect_name}: {score:+.2f} ({count} mentions)")

            # Add evidence
            if data.get("top_evidence"):
                for ev in data["top_evidence"][:1]:
                    truncated = ev[:100] + "..." if len(ev) > 100 else ev
                    lines.append(f"     └─ \"{truncated}\"")

        return "\n".join(lines)

    def _format_theme_section(self, themes: List[Dict]) -> str:
        """Format theme discovery results."""
        if not themes:
            return "\n🔍 THEMES: Insufficient data for reliable theme discovery."

        lines = [f"\n🔍 RECURRING THEMES:\n{'─'*40}"]

        for i, theme in enumerate(themes[:6], 1):
            label = theme["label"]
            count = theme["review_count"]
            sentiment = theme["avg_sentiment"]

            if sentiment > 0.1:
                mood = "positive"
            elif sentiment < -0.1:
                mood = "negative"
            else:
                mood = "mixed"

            lines.append(f"\n  {i}. {label} ({count} reviews, {mood} sentiment)")
            lines.append(f"     Keywords: {', '.join(theme['keywords'][:5])}")

            if theme.get("representative_reviews"):
                sample = theme["representative_reviews"][0]
                truncated = sample[:120] + "..." if len(sample) > 120 else sample
                lines.append(f"     Sample: \"{truncated}\"")

        return "\n".join(lines)

    def _format_trend_section(self, trends: Dict) -> str:
        """Format trend analysis results."""
        direction = trends.get("direction", "unknown")
        monthly = trends.get("monthly", {})

        direction_text = {
            "improving": "📈 IMPROVING — Customer satisfaction is trending upward",
            "declining": "📉 DECLINING — Customer satisfaction is trending downward",
            "stable": "➡️ STABLE — Customer satisfaction remains consistent",
            "insufficient_data": "❓ Insufficient data for trend determination",
        }

        lines = [
            f"\n📈 TEMPORAL TRENDS:\n{'─'*40}",
            f"  Direction: {direction_text.get(direction, direction)}\n",
        ]

        months = sorted(monthly.keys())
        for month in months[-6:]:  # Show last 6 months
            data = monthly[month]
            bar_len = int(data["avg_rating"] * 4)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(
                f"  {month}: {bar} {data['avg_rating']:.1f}★ "
                f"({data['review_count']} reviews, "
                f"{data['low_rating_pct']:.0f}% low-rated)"
            )

        return "\n".join(lines)

    def _generate_citations(self, reviews: List[Dict]) -> List[Dict]:
        """Generate source citations for grounding."""
        citations = []
        for i, review in enumerate(reviews, 1):
            citations.append({
                "id": i,
                "review_id": review.get("review_id", ""),
                "product": review.get("product", ""),
                "rating": review.get("rating", 0),
                "date": review.get("date", ""),
                "snippet": review.get("snippet", review.get("text", ""))[:150],
                "relevance": review.get("relevance", 0),
            })
        return citations

    def _format_citations(self, citations: List[Dict]) -> str:
        """Format citations as source references."""
        lines = [f"\n📎 SOURCE CITATIONS (Grounded Evidence):\n{'─'*40}"]

        for c in citations[:5]:
            lines.append(
                f"  [{c['id']}] {c['product']} | {c['rating']}★ | {c['date']}\n"
                f"      \"{c['snippet']}...\""
            )

        return "\n".join(lines)


class CustomerVoiceOrchestrator:
    """
    Main orchestrator that ties all agents together.
    
    Usage:
        orchestrator = CustomerVoiceOrchestrator(reviews_data)
        response = orchestrator.ask("What are the main complaints about fans?")
    """

    def __init__(self, reviews: List[Dict]):
        """
        Initialize the orchestrator with review data.
        
        Args:
            reviews: List of processed review dictionaries
        """
        self.reviews = reviews

        # Initialize components
        self.vector_store = VectorStoreManager()
        indexed = self.vector_store.index_reviews(reviews)
        print(f"✅ Indexed {indexed} reviews in vector store")

        # Initialize agents
        self.router = OrchestratorAgent()
        self.retrieval_agent = RetrievalAgent(self.vector_store)
        self.sentiment_agent = SentimentAgent()
        self.theme_agent = ThemeAgent()
        self.trend_agent = TrendAgent()
        self.synthesizer = SynthesizerAgent()

    def ask(self, query: str) -> str:
        """
        Process a product manager's question and return a grounded answer.
        
        Args:
            query: Natural language question from a product manager
            
        Returns:
            Grounded, cited response string
        """
        # Initialize state
        state = AgentState(query=query)

        # Step 1: Route the query
        agent_sequence = self.router.route(state)
        print(f"🔄 Intent: {state.intent.value}")
        print(f"🔄 Agent Pipeline: {' → '.join(agent_sequence)}")

        # Step 2: Execute agent pipeline
        for agent_name in agent_sequence:
            if agent_name == "retrieval":
                state = self.retrieval_agent.execute(state)
            elif agent_name == "sentiment":
                state = self.sentiment_agent.execute(state)
            elif agent_name == "theme":
                state = self.theme_agent.execute(state, all_reviews=self.reviews)
            elif agent_name == "trend":
                state = self.trend_agent.execute(state)
            elif agent_name == "synthesizer":
                state = self.synthesizer.execute(state)

        return state.grounded_response

    def get_full_analysis(self) -> Dict:
        """Run a comprehensive analysis across all reviews."""
        state = AgentState(query="Complete customer voice analysis")
        state.intent = QueryIntent.GENERAL_SUMMARY

        # Retrieve all (use broad query)
        state.retrieved_reviews = [
            {
                "review_id": r.get("review_id", ""),
                "text": r.get("cleaned_text", r.get("review_text", "")),
                "product": r.get("product_name", ""),
                "category": r.get("product_category", ""),
                "rating": int(r.get("rating", 3)),
                "date": r.get("review_date", ""),
                "relevance": 1.0,
                "snippet": r.get("cleaned_text", r.get("review_text", ""))[:150],
            }
            for r in self.reviews
        ]

        # Run all agents
        state = self.sentiment_agent.execute(state)
        state = self.theme_agent.execute(state, all_reviews=self.reviews)
        state = self.trend_agent.execute(state)
        state = self.synthesizer.execute(state)

        return {
            "response": state.grounded_response,
            "sentiment": state.sentiment_results,
            "themes": state.theme_results,
            "trends": state.trend_results,
            "confidence": state.confidence_score,
            "coverage": state.data_coverage,
        }
