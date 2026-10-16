
"""Resource limits for CiteWise Stage 1."""

MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_PDF_PAGES = 300

# A page with fewer than this many non-whitespace
# characters is flagged for possible OCR.
MIN_TEXT_CHARACTERS = 20
