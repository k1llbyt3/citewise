import pytest,json
from citewise.retriever import BM25Retriever, load_chunks, tokenize

def write_report(path, chunks):
    report = {
        "document": {
            "filename": "security_guide.pdf",
            "page_count": 2,
        },
        "chunks": chunks,
        "chunk_count": len(chunks),
        "status": "success",
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path



def make_chunk(chunk_id, page_number, text):
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-123",
        "page_number": page_number,
        "chunk_index": 1,
        "text": text,
        "extraction_method": "text",
        "filename": "security_guide.pdf",
    }


def test_tokenize_lowercases_and_splits_words():
    assert tokenize("PDF Security, SECURITY!") == [
        "pdf", "security", "security"
    ]


def test_load_chunks_preserves_source_metadata(tmp_path):
    chunk = make_chunk(
        "chunk-1", 2, "Use encryption to protect stored data."
    )
    path = write_report(tmp_path / "report.json", [chunk])

    chunks = load_chunks([path])

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "chunk-1"
    assert chunks[0]["document_id"] == "doc-123"
    assert chunks[0]["page_number"] == 2
    assert chunks[0]["filename"] == "security_guide.pdf"


def test_bm25_ranks_more_relevant_chunk_first():
    chunks = [
        make_chunk(
            "chunk-security",
            1,
            "Security controls protect the application.",
        ),
        make_chunk(
            "chunk-encryption",
            2,
            "Security and encryption protect stored data.",
        ),
        make_chunk(
            "chunk-cooking",
            3,
            "Cooking instructions for a vegetable soup.",
        ),
    ]

    retriever = BM25Retriever(chunks)
    results = retriever.search("security encryption", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-encryption"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["page_number"] == 2
    assert results[0]["filename"] == "security_guide.pdf"


def test_empty_or_unmatched_query_returns_no_results():
    chunks = [
        make_chunk("chunk-1", 1, "Security protects user data.")
    ]
    retriever = BM25Retriever(chunks)

    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert retriever.search("volcano") == []


def test_top_k_limits_number_of_results():
    chunks = [
        make_chunk("chunk-1", 1, "Security protects data."),
        make_chunk("chunk-2", 2, "Security protects accounts."),
        make_chunk("chunk-3", 3, "Security protects applications."),
    ]

    results = BM25Retriever(chunks).search("security", top_k=2)

    assert len(results) == 2


def test_invalid_top_k_is_rejected():
    retriever = BM25Retriever(
        [make_chunk("chunk-1", 1, "Security protects data.")]
    )

    with pytest.raises(ValueError, match="positive integer"):
        retriever.search("security", top_k=0)


def test_missing_report_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_chunks([tmp_path / "missing.json"])


def test_invalid_json_report_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_chunks([path])

def test_search_returns_empty_for_blank_query():
    retriever = BM25Retriever([
        make_chunk("chunk-1", 1, "Python is used for data analysis.")
    ])

    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_search_returns_empty_when_no_terms_match():
    retriever = BM25Retriever([
        make_chunk("chunk-1", 1, "Python is used for data analysis.")
    ])

    assert retriever.search("quantum astronomy") == []
