import pytest
from src.core.data_ingestion import DataIngestionPipeline


@pytest.fixture
def pipeline():
    return DataIngestionPipeline()

def test_clean_text(pipeline):
    # Test whitespace and URLs
    raw = "  Terrible product!! https://example.com  Call 999  "
    cleaned = pipeline.clean_text(raw)
    assert "https://example.com" not in cleaned
    assert "Terrible product!" in cleaned
    
def test_aspect_extraction(pipeline):
    text = "The motor is very noisy and power consumption is high."
    aspects = pipeline.extract_aspects(text)
    assert "noise" in aspects
    assert "energy_efficiency" in aspects

def test_sentiment_signals(pipeline):
    text1 = "This is the worst fan ever, totally defective."
    comp1, praise1 = pipeline.detect_sentiment_signals(text1)
    assert comp1 is True
    assert praise1 is False
    
    text2 = "Amazing quality, highly recommend it!"
    comp2, praise2 = pipeline.detect_sentiment_signals(text2)
    assert comp2 is False
    assert praise2 is True

def test_process_single_review(pipeline):
    raw = {
        "review_id": "123",
        "review_text": "The fan wobbles a lot and is very noisy. Regret buying it.",
        "product_name": "Havells Pacer",
        "product_category": "fans",
        "rating": 1,
    }
    
    processed = pipeline.process_single_review(raw)
    assert processed.review_id == "123"
    assert "noise" in processed.mentioned_aspects
    assert processed.has_complaint is True
    assert processed.word_count > 5
