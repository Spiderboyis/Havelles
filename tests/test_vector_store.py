import pytest
from src.core.vector_store import VectorStoreManager

@pytest.fixture
def store():
    return VectorStoreManager()

@pytest.fixture
def sample_reviews():
    return [
        {"review_id": "1", "review_text": "The mixer grinder stopped working. Defective motor makes a loud noise.", "product_category": "mixer_grinders", "rating": 1},
        {"review_id": "2", "review_text": "Excellent fan, very silent and good air delivery.", "product_category": "fans", "rating": 5},
        {"review_id": "3", "review_text": "Water heater is leaking from the bottom valve.", "product_category": "water_heaters", "rating": 2},
        {"review_id": "4", "review_text": "The fan makes a loud clicking noise at speed 5.", "product_category": "fans", "rating": 2},
    ]

def test_indexing(store, sample_reviews):
    count = store.index_reviews(sample_reviews)
    assert count == 4
    stats = store.get_stats()
    assert stats["total_documents"] == 4

def test_retrieval(store, sample_reviews):
    store.index_reviews(sample_reviews)
    
    # Query for fan noise
    results = store.retrieve("fan noise loud", top_k=2)
    assert len(results) > 0
    assert results[0].review_id == "4"
    assert "noise" in results[0].review_text.lower()
    
    # Query with category filter
    results_filtered = store.retrieve("water", category_filter="water_heaters")
    assert len(results_filtered) == 1
    assert results_filtered[0].review_id == "3"
