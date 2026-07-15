import pytest
from src.core.sentiment_engine import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

def test_document_sentiment_positive(analyzer):
    text = "This fan is absolutely amazing and perfect for my room."
    score, label = analyzer.analyze_document_sentiment(text)
    assert score > 0.5
    assert label in ["positive", "very_positive"]

def test_document_sentiment_negative(analyzer):
    text = "Terrible product, completely defective and a waste of money."
    score, label = analyzer.analyze_document_sentiment(text)
    assert score < -0.5
    assert label in ["negative", "very_negative"]

def test_negation_handling(analyzer):
    # Base positive word but negated
    text = "The build quality is not good."
    score, label = analyzer.analyze_document_sentiment(text)
    assert score < 0  # Should be negative due to "not"

def test_aspect_sentiment(analyzer):
    text = "The air delivery is excellent, but it is extremely noisy at high speeds."
    aspects = analyzer.analyze_aspect_sentiment(text)
    
    # We should have found performance and noise aspects
    aspect_dict = {a.aspect: a for a in aspects}
    
    assert "performance" in aspect_dict
    assert aspect_dict["performance"].sentiment_score > 0
    
    assert "noise" in aspect_dict
    assert aspect_dict["noise"].sentiment_score < 0
