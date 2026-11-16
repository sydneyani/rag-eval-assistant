from citeground.text import tokenize, windows


def test_tokenizer_keeps_amounts_and_hyphenated_names() -> None:
    tokens = tokenize("HIR-3 average selling price was $428,000, up 18.1 percent.")
    assert "hir-3" in tokens
    assert "428" in tokens or "428000" in tokens
    assert "18.1" in tokens
    assert "price" in tokens
    assert "was" not in tokens
    assert "the" not in tokens


def test_tokenizer_strips_currency_and_percent_markers() -> None:
    assert tokenize("$4.82 billion and 61.4%") == ["4.82", "billion", "61.4"]
    assert "4.82" in tokenize("Revenue was $4.82 billion")
    assert "61.4" in tokenize("margin was 61.4 percent")


def test_ten_k_token_stays_together() -> None:
    assert "10-k" in tokenize("This Form 10-K reports revenue")


def test_windows_cover_the_text_with_overlap() -> None:
    text = "a" * 2000
    spans = windows(text, 500, 100)
    assert spans == [
        (0, 500),
        (400, 900),
        (800, 1300),
        (1200, 1700),
        (1600, 2000),
    ]


def test_windows_reject_overlap_that_does_not_advance() -> None:
    try:
        windows("abc", 4, 4)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
