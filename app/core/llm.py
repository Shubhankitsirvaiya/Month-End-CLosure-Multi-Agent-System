"""LLM provider helpers for live dynamic orchestration."""

from __future__ import annotations

from app.core.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None):
    """Build the required chat model used for agent planning.

    Args:
        settings: Explicit settings are useful in tests. ``None`` loads the
            process-wide environment-backed settings.

    Returns:
        A configured ``ChatOpenAI`` client.

    Raises:
        RuntimeError: When OpenAI mode was requested without a key.
        ValueError: When the provider name is not supported.
    """
    # A caller-supplied Settings object is useful in tests; otherwise load .env.
    settings = settings or get_settings()
    # Detect a missing secret before an agent starts so failure is immediate and
    # actionable rather than appearing halfway through a review.
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is empty. Fill the key in the project .env file "
            "or process environment before starting the dynamic agents."
        )
    # Import lazily so configuration errors are reported before adapter creation.
    from langchain_openai import ChatOpenAI

    # Do not pass ``temperature``: GPT-5-family models accept only their supported
    # default. timeout bounds one network call; max_retries covers brief failures.
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        timeout=60,
        max_retries=2,
    )

