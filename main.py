#!/usr/bin/env python3
"""
Customer Voice Intelligence Agent — Main Runner

This script demonstrates the complete pipeline:
1. Generate synthetic review data (or load existing)
2. Ingest and preprocess reviews
3. Initialize the multi-agent system
4. Run sample queries (interactive and demo mode)
5. Generate evaluation metrics
6. Generate the solution PDF report

Usage:
    python main.py              # Run full demo
    python main.py --interactive # Interactive Q&A mode
    python main.py --pdf-only   # Only generate the PDF report
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.review_generator import generate_dataset
from src.core.data_ingestion import DataIngestionPipeline
from src.core.sentiment_engine import SentimentAnalyzer
from src.core.theme_engine import ThemeDiscoveryEngine
from src.core.evaluation import EvaluationFramework, TEST_QUERIES
from src.agents.orchestrator import CustomerVoiceOrchestrator


def print_banner():
    """Print the application banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     🎯  CUSTOMER VOICE INTELLIGENCE AGENT  🎯               ║
║     Havells Consumer Products — AI Catalyst Program          ║
║     Set A: Customer Voice Intelligence                       ║
╚══════════════════════════════════════════════════════════════╝
    """)


def step_1_generate_data(data_dir: Path, num_reviews: int = 2500) -> list:
    """Step 1: Generate or load synthetic review data."""
    print("\n" + "="*60)
    print("📦 STEP 1: Data Generation & Loading")
    print("="*60)

    json_path = data_dir / "havells_reviews.json"

    if json_path.exists():
        print(f"  📂 Loading existing dataset from {json_path}")
        with open(json_path, "r") as f:
            reviews = json.load(f)
        print(f"  ✅ Loaded {len(reviews)} reviews")
    else:
        print(f"  🔧 Generating {num_reviews} synthetic reviews...")
        reviews = generate_dataset(
            num_reviews=num_reviews,
            start_date="2024-01-01",
            end_date="2025-06-30",
            output_path=data_dir
        )

    return reviews


def step_2_ingest_data(raw_reviews: list) -> list:
    """Step 2: Ingest and preprocess reviews."""
    print("\n" + "="*60)
    print("🔄 STEP 2: Data Ingestion & Preprocessing")
    print("="*60)

    pipeline = DataIngestionPipeline()
    processed = pipeline.process_batch(raw_reviews, deduplicate=True)

    stats = pipeline.get_ingestion_stats(processed)
    print(f"  ✅ Processed {stats['total_reviews']} reviews")
    print(f"  📊 Avg Rating: {stats['avg_rating']:.2f}")
    print(f"  📊 Complaint %: {stats['complaint_percentage']}%")
    print(f"  📊 Praise %: {stats['praise_percentage']}%")
    print(f"  📊 Categories: {json.dumps(stats['categories'], indent=6)}")
    print(f"  📊 Date Range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")

    # Convert back to dict format for downstream components
    review_dicts = [r.to_dict() for r in processed]
    return review_dicts


def step_3_run_analysis(orchestrator: CustomerVoiceOrchestrator) -> dict:
    """Step 3: Run the full analysis pipeline."""
    print("\n" + "="*60)
    print("🧠 STEP 3: Multi-Agent Analysis")
    print("="*60)

    print("  Running comprehensive analysis...")
    analysis = orchestrator.get_full_analysis()

    print("\n" + analysis["response"])

    return analysis


def step_4_demo_queries(orchestrator: CustomerVoiceOrchestrator) -> list:
    """Step 4: Run demo queries to showcase the system."""
    print("\n" + "="*60)
    print("💬 STEP 4: Sample Query Demonstrations")
    print("="*60)

    demo_queries = [
        "What are the main complaints about Havells ceiling fans?",
        "How has customer satisfaction with water heaters changed over time?",
        "What do customers think about the noise levels of Havells products?",
        "Tell me about after-sales service complaints across all categories",
        "Which product category has the best customer sentiment?",
        "Are quality issues getting better or worse over the last year?",
    ]

    results = []
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'─'*60}")
        print(f"  ❓ Query {i}: {query}")
        print(f"{'─'*60}")

        response = orchestrator.ask(query)
        print(response)
        results.append({"query": query, "response": response})

    return results


def step_5_evaluate(orchestrator: CustomerVoiceOrchestrator, review_dicts: list) -> dict:
    """Step 5: Run evaluation metrics."""
    print("\n" + "="*60)
    print("📏 STEP 5: System Evaluation")
    print("="*60)

    evaluator = EvaluationFramework()

    # Sentiment accuracy evaluation
    analyzer = SentimentAnalyzer()
    predicted_labels = []
    true_labels = []

    sample_size = min(500, len(review_dicts))
    for review in review_dicts[:sample_size]:
        text = review.get("cleaned_text", review.get("review_text", ""))
        _, pred_label = analyzer.analyze_document_sentiment(text)
        predicted_labels.append(pred_label)
        true_labels.append(review.get("sentiment_label", "neutral"))

    sentiment_eval = evaluator.evaluate_sentiment_accuracy(predicted_labels, true_labels)
    print(f"\n  📊 Sentiment Accuracy: {sentiment_eval['ternary_accuracy']:.1%}")
    for cls, metrics in sentiment_eval.get("per_class", {}).items():
        print(f"     {cls}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")

    # Theme coherence evaluation
    # Run theme discovery for evaluation
    theme_engine = ThemeDiscoveryEngine(n_themes=10, min_theme_size=5)
    texts = [r.get("cleaned_text", r.get("review_text", "")) for r in review_dicts]
    ids = [r.get("review_id", str(i)) for i, r in enumerate(review_dicts)]
    sentiments = [0.0] * len(review_dicts)
    categories = [r.get("product_category", "unknown") for r in review_dicts]
    dates = [r.get("review_date", "2024-01-01") for r in review_dicts]

    themes = theme_engine.discover_themes(texts, ids, sentiments, categories, dates)
    theme_dicts = [
        {"label": t.label, "keywords": t.keywords, "review_count": t.review_count}
        for t in themes
    ]
    theme_eval = evaluator.evaluate_theme_coherence(theme_dicts)
    print(f"\n  🔍 Theme Coherence:")
    print(f"     Themes discovered: {theme_eval['num_themes']}")
    print(f"     Distinctness: {theme_eval['theme_distinctness']:.2f}")
    print(f"     Size uniformity: {theme_eval['size_uniformity']:.2f}")

    # Full system test
    system_eval = evaluator.run_full_evaluation(orchestrator, TEST_QUERIES)
    print(f"\n  🧪 System Tests:")
    print(f"     Queries tested: {system_eval['queries_tested']}")
    print(f"     All structured: {system_eval['all_structured']}")
    print(f"     All grounded: {system_eval['all_mention_data']}")

    return {
        "sentiment": sentiment_eval,
        "theme": theme_eval,
        "system": system_eval,
    }


def run_interactive_mode(orchestrator: CustomerVoiceOrchestrator):
    """Run interactive Q&A mode."""
    print("\n" + "="*60)
    print("💬 INTERACTIVE MODE")
    print("  Type your questions about Havells products.")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*60)

    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break
            if not query:
                continue

            response = orchestrator.ask(query)
            print(response)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Customer Voice Intelligence Agent")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--pdf-only", action="store_true", help="Only generate PDF report")
    parser.add_argument("--reviews", type=int, default=2500, help="Number of reviews to generate")
    args = parser.parse_args()

    print_banner()

    data_dir = PROJECT_ROOT / "data"

    if args.pdf_only:
        print("📄 Generating PDF report only...")
        from src.utils.pdf_generator import generate_solution_pdf
        generate_solution_pdf(PROJECT_ROOT / "output" / "Customer_Voice_Intelligence_Agent_Solution.pdf")
        print("✅ PDF generated!")
        return

    # Step 1: Data
    raw_reviews = step_1_generate_data(data_dir, args.reviews)

    # Step 2: Ingest
    review_dicts = step_2_ingest_data(raw_reviews)

    # Step 3: Initialize orchestrator
    print("\n" + "="*60)
    print("🤖 Initializing Multi-Agent System...")
    print("="*60)
    orchestrator = CustomerVoiceOrchestrator(review_dicts)

    if args.interactive:
        run_interactive_mode(orchestrator)
    else:
        # Step 3: Full analysis
        analysis = step_3_run_analysis(orchestrator)

        # Step 4: Demo queries
        demo_results = step_4_demo_queries(orchestrator)

        # Step 5: Evaluation
        eval_results = step_5_evaluate(orchestrator, review_dicts)

        # Step 6: Generate PDF
        print("\n" + "="*60)
        print("📄 STEP 6: Generating Solution PDF Report")
        print("="*60)
        from src.utils.pdf_generator import generate_solution_pdf
        pdf_path = PROJECT_ROOT / "output" / "Customer_Voice_Intelligence_Agent_Solution.pdf"
        generate_solution_pdf(pdf_path, eval_results)
        print(f"  ✅ PDF saved to: {pdf_path}")

        print("\n" + "="*60)
        print("🎉 ALL STEPS COMPLETE!")
        print("="*60)
        print(f"  📄 Solution PDF: {pdf_path}")
        print(f"  📊 Data: {data_dir}")
        print(f"  💡 Run 'python main.py --interactive' for Q&A mode")


if __name__ == "__main__":
    main()
