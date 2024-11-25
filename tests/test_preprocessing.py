# tests/test_preprocessing.py
from src.preprocessing import preprocess_text

def test_preprocess_text():
    text = "Hello @user! Visit https://example.com #AI"
    expected = "hello visit ai"
    assert preprocess_text(text) == expected
