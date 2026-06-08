# ===================================================================
# Turns a model verdict + an inspection photo into a short written
# report using Gemini (vision) grounded with notes retrieved from the
# local knowledge base in knowledge_base.py (a small RAG pipeline:
# retrieve relevant notes -> stuff them into the prompt -> generate).
# ===================================================================

import os
import time

from google import genai
from google.genai import types

from knowledge_base import retrieve

# Try the full model first; gemini-2.5-flash occasionally returns a transient
# 503 "model overloaded" error, so each model gets a couple of retries before
# we fall back to the lighter -lite model.
REPORT_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
RETRIES_PER_MODEL = 2
RETRY_DELAY_SECONDS = 3

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _generate_with_retry(contents):
    """Call Gemini, retrying transient errors and falling back across models."""
    last_error = None
    for model in REPORT_MODELS:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                return _client.models.generate_content(model=model, contents=contents)
            except Exception as exc:
                last_error = exc
                if attempt < RETRIES_PER_MODEL - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
    raise last_error


def _build_query(label, probability):
    if label.lower().startswith("crack"):
        return (
            "crack detected in concrete surface high probability structural "
            "severity cause remediation"
        )
    return (
        "no crack detected concrete surface routine maintenance "
        "preventive inspection"
    )


def generate_report(image_path, label, probability):
    """Generate a short professional crack-analysis report for one inspection.

    Retrieves grounding notes from the local knowledge base, then asks
    Gemini to write the report from those notes plus the photo and the
    model's verdict. Returns the report text, or a fallback message if
    the LLM call fails (so a flaky API never breaks the whole request).
    """
    notes = retrieve(_build_query(label, probability), top_k=4)
    notes_block = "\n".join(f"- {note}" for note in notes)

    prompt = f"""You are a structural-inspection assistant helping a property
owner understand an automated crack-detection result.

Automated model verdict: "{label}" (crack probability: {probability:.1%})

Relevant notes from the inspection knowledge base (use these as your
factual grounding -- do not contradict them):
{notes_block}

Look at the attached photo of the concrete surface and write a short
report for a non-expert reader with exactly these sections:
1. Summary - one or two sentences on what was found.
2. Likely cause - the most plausible explanation given the photo and verdict.
3. Severity - low / moderate / high, with a one-line justification.
4. Recommended action - concrete next steps the owner should take.

Keep it under 180 words total, plain text, no markdown headers or
asterisks -- use the section names followed by a colon."""

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = _generate_with_retry([
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ])
        return response.text.strip()
    except Exception as exc:
        return (
            "Automatic report generation is temporarily unavailable "
            f"({type(exc).__name__}). The detection result above is still valid; "
            "please try again in a moment for a full written report."
        )
