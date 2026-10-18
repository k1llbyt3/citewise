"""Generate text embeddings using the Gemini API."""
import os
from collections.abc import Sequence
from dotenv import load_dotenv
from google import genai
from google.genai import types
from citewise.api_limits import wait_for_api_slot

MODEL_NAME = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
DEFAULT_BATCH_SIZE = 32

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"


def _get_client() -> genai.Client:
    """Create an API client using the key in the environment or .env file."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to the project .env file."
        )

    return genai.Client(api_key=api_key)


def embed_texts(
    texts: Sequence[str],
    task_type: str = DOCUMENT_TASK,
    batch_size: int = DEFAULT_BATCH_SIZE,
    client=None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts in batches.

    Args:
        texts: Text passages or search queries to embed.
        task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY.
        batch_size: Maximum number of texts sent in one API request.
        client: Optional Gemini client, useful for testing without API calls.

    Returns:
        One embedding vector per input text, in the original order.
    """
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not one string.")

    if not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of strings.")

    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every item in texts must be a string.")

    if any(not text.strip() for text in texts):
        raise ValueError("Texts must not contain empty or whitespace-only items.")

    if task_type not in {DOCUMENT_TASK, QUERY_TASK}:
        raise ValueError(
            f"Unsupported task_type: {task_type}. "
            f"Use {DOCUMENT_TASK} or {QUERY_TASK}."
        )

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")

    if not texts:
        return []

    if client is None:
        client = _get_client()

    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        wait_for_api_slot()
        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

        if response.embeddings is None:
            raise RuntimeError("Gemini returned no embeddings.")

        if len(response.embeddings) != len(batch):
            raise RuntimeError(
                "Gemini returned a different number of embeddings "
                "than the number of input texts."
            )

        for item in response.embeddings:
            if item.values is None:
                raise RuntimeError("Gemini returned an empty embedding vector.")

            vector = [float(value) for value in item.values]

            if len(vector) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
                    f"but received {len(vector)}."
                )

            all_embeddings.append(vector)

    return all_embeddings


def embed_document(text: str, client=None) -> list[float]:
    """Generate one embedding for a document chunk."""
    return embed_texts(
        [text],
        task_type=DOCUMENT_TASK,
        client=client,
    )[0]


def embed_query(query: str, client=None) -> list[float]:
    """Generate one embedding for a search query."""
    return embed_texts(
        [query],
        task_type=QUERY_TASK,
        client=client,
    )[0]
