"""Domain tagging — LLM-assigned tags from a controlled vocabulary."""

import logging
from typing import Optional

from quantmind.llm_client import LLMClient, ProviderUnavailableError

logger = logging.getLogger("quantmind.ingest.tagging")

CONTROLLED_VOCABULARY = [
    "factor-models",
    "risk-management",
    "nlp-finance",
    "portfolio-optimization",
    "market-microstructure",
    "alternative-data",
    "ml-methods",
    "macro",
    "sentiment",
]

TAGGING_PROMPT = """You are a financial domain classifier. Given a document title
and abstract, assign 1-3 tags from this controlled vocabulary:

{fixed_vocabulary}

Return ONLY a JSON array of strings, nothing else. Example: ["factor-models", "ml-methods"]

Title: {title}
Abstract: {abstract}

Tags:"""


def tag_document(
    title: str,
    abstract: str,
    llm: Optional[LLMClient] = None,
) -> list[str]:
    """Assign 1-3 domain tags to a document using the LLM.

    Args:
        title: Document title.
        abstract: Document abstract / summary.
        llm: LLMClient instance. A new one is created if not provided.

    Returns:
        A list of 1-3 tags from the controlled vocabulary.
    """
    if not title and not abstract:
        logger.warning("No content to tag — returning empty tags")
        return []

    own_client = False
    if llm is None:
        try:
            llm = LLMClient()
            own_client = True
        except Exception:
            return []

    try:
        prompt = TAGGING_PROMPT.format(
            fixed_vocabulary="\n".join(f"  - {t}" for t in CONTROLLED_VOCABULARY),
            title=title or "(no title)",
            abstract=abstract or "(no abstract)",
        )
        response = llm.complete(
            system="You are a precise classifier. Output only the JSON array.",
            user=prompt,
        )
        tags = _parse_tags_response(response)
        return tags
    except ProviderUnavailableError:
        logger.warning("LLM unavailable — skipping domain tagging")
        return []
    except Exception:
        logger.exception("Domain tagging failed")
        return []
    finally:
        if own_client and llm:
            pass  # No close needed


def _parse_tags_response(response: str) -> list[str]:
    """Parse the LLM response into a list of valid tags."""
    import json
    import re

    # Try to extract a JSON array from the response
    match = re.search(r"\[.*?\]", response.strip(), re.DOTALL)
    if match:
        try:
            raw = json.loads(match.group())
            if isinstance(raw, list):
                return [t for t in raw if t in CONTROLLED_VOCABULARY]
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: look for known tags by name
    found = []
    for tag in CONTROLLED_VOCABULARY:
        if tag.lower() in response.lower():
            found.append(tag)
    return found[:3]
