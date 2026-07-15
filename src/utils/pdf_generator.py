"""PDF Solution Report Generator using fpdf2."""
from fpdf import FPDF
from pathlib import Path


class SolutionPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Havells Customer Voice Intelligence Agent | AI Catalyst Program", align="C")
        self.ln(10)
        self.set_draw_color(0, 102, 204)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 51, 102)
        self.ln(6)
        self.cell(0, 10, title)
        self.ln(8)
        self.set_draw_color(0, 102, 204)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(6)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(51, 51, 51)
        self.cell(0, 8, title)
        self.ln(7)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 33, 33)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 33, 33)
        x = self.get_x()
        self.cell(8, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font("Courier", "", 8)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(33, 33, 33)
        self.multi_cell(0, 4.5, text, fill=True)
        self.ln(3)

    def table_row(self, cols, widths, bold=False):
        self.set_font("Helvetica", "B" if bold else "", 9)
        if bold:
            self.set_fill_color(0, 102, 204)
            self.set_text_color(255, 255, 255)
        else:
            self.set_fill_color(245, 245, 245)
            self.set_text_color(33, 33, 33)
        for i, (col, w) in enumerate(zip(cols, widths)):
            self.cell(w, 7, str(col), border=1, fill=True, align="C" if i > 0 else "L")
        self.ln()


def generate_solution_pdf(output_path, eval_results=None):
    pdf = SolutionPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── COVER PAGE ──
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 15, "Customer Voice", align="C")
    pdf.ln(14)
    pdf.cell(0, 15, "Intelligence Agent", align="C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Set A - AI Catalyst Program | Havells India Ltd.", align="C")
    pdf.ln(12)
    pdf.set_draw_color(0, 102, 204)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(66, 66, 66)
    pdf.cell(0, 8, "An Agentic AI System for Automated Review Analysis,", align="C")
    pdf.ln(7)
    pdf.cell(0, 8, "Theme Discovery, and Grounded Product Intelligence", align="C")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, "Built with: Python | Scikit-learn | Multi-Agent Architecture | RAG", align="C")
    pdf.ln(15)
    pdf.set_font("Helvetica", "U", 11)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 8, "GitHub Repository: https://github.com/Spiderboyis/Havelles", align="C", link="https://github.com/Spiderboyis/Havelles")

    # ── 1. EXECUTIVE SUMMARY ──
    pdf.add_page()
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "This document presents a production-grade agentic AI system that transforms raw, messy "
        "customer reviews of Havells consumer appliances into actionable product intelligence. "
        "The system automatically identifies recurring complaint themes, tracks sentiment shifts "
        "over time, and answers product manager queries in plain English - with every answer "
        "grounded in actual review data."
    )
    pdf.body_text(
        "Key capabilities: (1) Aspect-Based Sentiment Analysis across 10 product dimensions, "
        "(2) Unsupervised theme discovery via TF-IDF + KMeans clustering, (3) Temporal trend "
        "detection with seasonal pattern awareness, (4) Grounded Q&A with source citations, "
        "(5) Multi-agent orchestration with intent-based routing."
    )
    pdf.sub_title("Problem Addressed")
    pdf.body_text(
        "Havells produces fans, water heaters, air purifiers, mixer grinders, and lighting products. "
        "Customer reviews pile up faster than teams can read. Product managers need to know: "
        "What are people unhappy about? On which product? Is it getting better or worse? "
        "This system answers these questions automatically, honestly, and with evidence."
    )

    # ── 2. SYSTEM ARCHITECTURE ──
    pdf.add_page()
    pdf.section_title("2. System Architecture")
    pdf.body_text(
        "The system follows the Orchestrator-Worker multi-agent pattern. A central Orchestrator "
        "agent receives the manager's question, classifies intent, and routes to specialist agents. "
        "Results are aggregated by a Synthesizer agent that ensures grounding."
    )
    pdf.sub_title("Architecture Diagram (Text)")
    pdf.code_block(
        "  Manager Query\n"
        "       |\n"
        "  [Orchestrator Agent] -- classifies intent, extracts filters\n"
        "       |\n"
        "  +----+-----+----------+-----------+\n"
        "  |          |          |           |\n"
        "  v          v          v           v\n"
        "[Retrieval] [Sentiment] [Theme]  [Trend]\n"
        "  Agent      Agent      Agent     Agent\n"
        "  |          |          |           |\n"
        "  +----+-----+----------+-----------+\n"
        "       |\n"
        "  [Synthesizer Agent] -- grounds & cites\n"
        "       |\n"
        "  Grounded Response + Citations"
    )
    pdf.sub_title("Agent Descriptions")
    agents = [
        ("Orchestrator", "Intent classification via regex patterns. Routes to specialists. No LLM needed."),
        ("Retrieval Agent", "TF-IDF vector search over indexed reviews. Supports filtering."),
        ("Sentiment Agent", "Lexicon-based ABSA with negation handling, intensifiers, emotion detection."),
        ("Theme Agent", "TF-IDF + KMeans clustering for unsupervised theme discovery."),
        ("Trend Agent", "Temporal aggregation computing monthly sentiment trends."),
        ("Synthesizer", "Aggregates outputs, validates grounding, generates cited response."),
    ]
    w = [40, 150]
    pdf.table_row(["Agent", "Role"], w, bold=True)
    for name, desc in agents:
        pdf.table_row([name, desc], w)

    # ── 3. DATA PIPELINE ──
    pdf.add_page()
    pdf.section_title("3. Data Pipeline")
    pdf.sub_title("3.1 Data Generation")
    pdf.body_text(
        "We built a synthetic review generator producing 2,500 realistic reviews across 5 product "
        "categories (fans 30%, water heaters 22%, mixer grinders 20%, lighting 16%, air purifiers 12%). "
        "Reviews include temporal patterns (seasonal complaint spikes), product-specific issues, "
        "varied writing styles, and Indian English patterns."
    )
    pdf.sub_title("3.2 Ingestion Pipeline")
    pdf.bullet("Text cleaning: URL/email removal, punctuation normalization, whitespace cleanup")
    pdf.bullet("Feature extraction: Aspect keyword detection across 10 product dimensions")
    pdf.bullet("Signal detection: Complaint/praise keyword matching for pre-filtering")
    pdf.bullet("Deduplication: MD5 hash-based duplicate removal")
    pdf.bullet("Validation: Minimum word count filtering, encoding normalization")

    pdf.sub_title("3.3 Data Flow")
    pdf.code_block(
        "Raw Reviews (JSON/CSV)\n"
        "  -> DataIngestionPipeline.process_batch()\n"
        "     -> clean_text() -> extract_aspects() -> detect_signals()\n"
        "  -> ProcessedReview objects\n"
        "  -> VectorStoreManager.index_reviews()\n"
        "  -> TF-IDF Matrix (ready for retrieval)"
    )

    # ── 4. CORE COMPONENTS ──
    pdf.add_page()
    pdf.section_title("4. Core Components Deep-Dive")
    pdf.sub_title("4.1 Sentiment Analysis Engine")
    pdf.body_text(
        "Hybrid approach combining domain-specific lexicon (80+ positive/negative terms with "
        "scores from -1 to +1) with context-aware features:"
    )
    pdf.bullet("Negation handling: 'not good' correctly flips to negative (partial inversion at 0.75x)")
    pdf.bullet("Intensifiers/diminishers: 'very good' = 1.3x, 'slightly bad' = 0.7x")
    pdf.bullet("Aspect-Based SA: 10 aspects (performance, noise, service, safety, etc.) with aspect-specific context words")
    pdf.bullet("Emotion detection: Frustration, satisfaction, anger, trust, disappointment")
    pdf.body_text(
        "Design Decision: We use lexicon-based rather than LLM-based sentiment because it provides "
        "deterministic, reproducible, auditable results at batch speed. LLM is reserved for the "
        "synthesis layer where natural language generation quality matters."
    )

    pdf.sub_title("4.2 Theme Discovery Engine")
    pdf.body_text(
        "Unsupervised theme extraction using TF-IDF vectorization + KMeans clustering:"
    )
    pdf.bullet("TF-IDF with 3000 features, (1,2)-gram range, sublinear TF weighting")
    pdf.bullet("KMeans with 10-12 clusters, 10 random initializations")
    pdf.bullet("Auto-labeling: Maps top cluster keywords to 16 predefined domain theme labels")
    pdf.bullet("Per-theme metrics: Sentiment distribution, category breakdown, temporal trend")
    pdf.bullet("Representative review selection via cosine similarity to cluster centroid")

    pdf.sub_title("4.3 Vector Store & Retrieval")
    pdf.body_text(
        "Lightweight TF-IDF vector store for grounded retrieval. Reviews are enriched with "
        "metadata (product name, category, rating-based sentiment words) before indexing. "
        "Retrieval supports category, rating range, and date filters. For production at scale, "
        "this swaps to ChromaDB or FAISS with sentence-transformer embeddings."
    )

    # ── 5. GROUNDING MECHANISM ──
    pdf.add_page()
    pdf.section_title("5. Grounding & Faithfulness")
    pdf.body_text(
        "The non-negotiable requirement: every answer must be grounded in actual review data. "
        "Our grounding strategy operates at three levels:"
    )
    pdf.sub_title("Level 1: Retrieval Grounding")
    pdf.body_text(
        "All analysis operates ONLY on retrieved reviews. The system never generates insights "
        "from parametric knowledge. If fewer than 5 reviews are retrieved, a data limitation "
        "warning is appended to the response."
    )
    pdf.sub_title("Level 2: Evidence Citations")
    pdf.body_text(
        "Every response includes SOURCE CITATIONS with review ID, product name, rating, date, "
        "and a verbatim snippet. Users can trace any claim back to specific reviews."
    )
    pdf.sub_title("Level 3: Honest Uncertainty")
    pdf.body_text(
        "The system explicitly states data coverage (e.g., 'Based on 20 relevant reviews'). "
        "When data is insufficient for a conclusion, it says so rather than inventing one. "
        "A confidence score (0-1) is computed based on retrieval coverage."
    )
    pdf.sub_title("Faithfulness Evaluation")
    pdf.body_text(
        "The EvaluationFramework checks: (1) Citation presence, (2) Data coverage mention, "
        "(3) Limitation acknowledgment, (4) Quote grounding rate - verifying quoted text "
        "against source reviews. These produce an overall faithfulness score."
    )

    # ── 6. QUERY PROCESSING ──
    pdf.add_page()
    pdf.section_title("6. Query Processing & Intent Classification")
    pdf.body_text(
        "The Orchestrator classifies queries into 7 intent categories using regex pattern matching "
        "(no LLM needed for routing, keeping it fast and deterministic):"
    )
    intents = [
        ("Sentiment Overview", "'How do customers feel about fans?'", "Retrieval -> Sentiment -> Synthesizer"),
        ("Theme Discovery", "'What are main complaints?'", "Retrieval -> Theme -> Synthesizer"),
        ("Trend Analysis", "'Is satisfaction improving?'", "Retrieval -> Sentiment -> Trend -> Synthesizer"),
        ("Product Comparison", "'Compare fans vs heaters'", "Retrieval -> Sentiment -> Theme -> Synthesizer"),
        ("Specific Issue", "'Tell me about noise issues'", "Retrieval -> Sentiment -> Theme -> Synthesizer"),
        ("Aspect Deep-Dive", "'How is build quality?'", "Retrieval -> Sentiment -> Synthesizer"),
        ("General Summary", "'Give me an overview'", "All agents"),
    ]
    w2 = [32, 50, 108]
    pdf.table_row(["Intent", "Example", "Agent Pipeline"], w2, bold=True)
    for intent, example, pipeline in intents:
        pdf.table_row([intent, example, pipeline], w2)

    pdf.sub_title("Filter Extraction")
    pdf.body_text(
        "The Orchestrator also extracts contextual filters from the query: product category "
        "(fans, water_heaters, etc.), aspect (noise, service, quality), and date ranges. "
        "These filters are passed to the Retrieval Agent for focused search."
    )

    # ── 7. EVALUATION ──
    pdf.add_page()
    pdf.section_title("7. Evaluation Framework")
    pdf.body_text("The system is evaluated across four dimensions:")

    pdf.sub_title("7.1 Retrieval Quality")
    pdf.body_text(
        "Category-level precision: For category-specific queries, what percentage of retrieved "
        "reviews belong to the correct category. Target: >80%."
    )
    pdf.sub_title("7.2 Sentiment Accuracy")
    pdf.body_text(
        "Ternary classification accuracy (positive/neutral/negative) measured against synthetic "
        "data ground truth. Per-class precision, recall, and F1 scores are computed."
    )
    if eval_results and "sentiment" in eval_results:
        s = eval_results["sentiment"]
        pdf.body_text(f"Measured Accuracy: {s.get('ternary_accuracy', 'N/A')}")

    pdf.sub_title("7.3 Theme Coherence")
    pdf.body_text(
        "Theme distinctness (1 - avg Jaccard overlap between keyword sets) and size uniformity "
        "(entropy-based measure of cluster balance). High distinctness = non-overlapping themes."
    )
    pdf.sub_title("7.4 Grounding Faithfulness")
    pdf.body_text(
        "Citation presence, data coverage mention, limitation acknowledgment, and quote "
        "grounding rate. Combined into an overall faithfulness score (0-1)."
    )

    # ── 8. SCALABILITY ──
    pdf.add_page()
    pdf.section_title("8. Scaling to Full Catalogue")
    pdf.body_text(
        "The problem statement asks: what would change if watching the whole catalogue? "
        "Here is our scaling roadmap:"
    )
    pdf.sub_title("Data Layer")
    pdf.bullet("Replace TF-IDF vectors with sentence-transformer dense embeddings (384-dim)")
    pdf.bullet("Swap in-memory store for ChromaDB/Pinecone with metadata filtering")
    pdf.bullet("Add Apache Kafka / Redis Streams for real-time review ingestion")
    pdf.bullet("Implement incremental indexing (add new reviews without full re-index)")

    pdf.sub_title("Compute Layer")
    pdf.bullet("Parallelize agent execution (sentiment + theme can run concurrently)")
    pdf.bullet("Cache frequent queries and pre-compute category-level summaries daily")
    pdf.bullet("Use async processing for LLM calls when synthesis uses GPT-4")

    pdf.sub_title("Agent Layer")
    pdf.bullet("Add a Scraping Agent for automated review collection from Amazon/Flipkart")
    pdf.bullet("Add an Alerting Agent for detecting sudden sentiment drops (anomaly detection)")
    pdf.bullet("Add a Comparison Agent for competitive analysis (Crompton, Bajaj, Orient)")
    pdf.bullet("Implement LangGraph for stateful multi-turn conversations with checkpointing")

    pdf.sub_title("Quality Layer")
    pdf.bullet("A/B testing framework for comparing analysis approaches")
    pdf.bullet("Human-in-the-loop feedback for theme label validation")
    pdf.bullet("Continuous evaluation dashboard tracking all 4 quality dimensions")

    # ── 9. TECH STACK ──
    pdf.add_page()
    pdf.section_title("9. Technology Stack")
    stack = [
        ("Language", "Python 3.10+"),
        ("NLP/ML", "Scikit-learn (TF-IDF, KMeans), NumPy"),
        ("Sentiment", "Custom lexicon engine with ABSA"),
        ("Topic Modeling", "TF-IDF + KMeans (BERTopic-ready)"),
        ("Vector Search", "TF-IDF cosine similarity (ChromaDB-ready)"),
        ("Agent Framework", "Custom orchestrator (LangGraph-ready)"),
        ("LLM (optional)", "OpenAI GPT-4o-mini for synthesis"),
        ("PDF Generation", "fpdf2"),
        ("Data Format", "JSON, CSV"),
    ]
    w3 = [50, 140]
    pdf.table_row(["Component", "Technology"], w3, bold=True)
    for comp, tech in stack:
        pdf.table_row([comp, tech], w3)

    pdf.ln(8)
    pdf.section_title("10. Project Structure")
    pdf.code_block(
        "havells/\n"
        "  main.py                    # Entry point\n"
        "  requirements.txt           # Dependencies\n"
        "  config/settings.py         # Centralized config\n"
        "  src/\n"
        "    agents/orchestrator.py   # Multi-agent system\n"
        "    core/\n"
        "      data_ingestion.py      # Preprocessing pipeline\n"
        "      sentiment_engine.py    # ABSA engine\n"
        "      theme_engine.py        # Topic discovery\n"
        "      vector_store.py        # Retrieval system\n"
        "      evaluation.py          # Quality metrics\n"
        "    data/review_generator.py # Synthetic data\n"
        "    utils/pdf_generator.py   # This report\n"
        "  data/                      # Generated datasets\n"
        "  output/                    # Reports & PDFs"
    )

    # ── 11. KEY DESIGN DECISIONS ──
    pdf.section_title("11. Key Design Decisions")
    decisions = [
        ("Lexicon over LLM for sentiment",
         "Deterministic, fast, auditable. LLM reserved for synthesis only."),
        ("TF-IDF over transformers for retrieval",
         "Zero GPU dependency. Swappable to dense embeddings for production."),
        ("Regex intent classification",
         "Fast, transparent routing. No LLM latency for query classification."),
        ("Stateless agents with explicit state",
         "Each agent is a pure function. No hidden state, easy to test and debug."),
        ("Synthetic data with realistic patterns",
         "Seasonal trends, Indian English, product-specific issues enable proper evaluation."),
        ("Grounding-first architecture",
         "Every response cites sources. System admits uncertainty rather than hallucinating."),
    ]
    for title, reason in decisions:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, title)
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, f"  Rationale: {reason}")
        pdf.ln(2)

    # ── CONCLUSION ──
    pdf.add_page()
    pdf.section_title("12. Conclusion")
    pdf.body_text(
        "This Customer Voice Intelligence Agent demonstrates a practical, production-oriented "
        "approach to automated review analysis. Unlike polished demos that hide complexity, "
        "this solution exposes the full design thinking: how data flows through the system, "
        "how work is split across specialized agents, how information passes between components, "
        "how output honesty is enforced through grounding, how quality is measured through "
        "a four-dimensional evaluation framework, and what would need to change at catalogue scale."
    )
    pdf.body_text(
        "The system processes 2,500 reviews across 5 Havells product categories, discovers "
        "recurring themes without labeled data, tracks sentiment trends over 18 months, and "
        "answers product manager questions with cited evidence. Every claim is traceable to "
        "source reviews, and every limitation is acknowledged."
    )
    pdf.body_text(
        "The architecture is designed for incremental enhancement: swap TF-IDF for transformers, "
        "add LangGraph for multi-turn conversations, integrate real-time scraping - each change "
        "is isolated to a single component while the agent coordination layer remains stable."
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"PDF saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    generate_solution_pdf("output/Customer_Voice_Intelligence_Agent_Solution.pdf")
