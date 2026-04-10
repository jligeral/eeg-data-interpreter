"""
reasoning_agent/agent.py — drive Qwen3:8b via Ollama to reason over EEG features.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from feature_extractor.feature_record import FeatureRecord
from reasoning_agent.embedding_analysis import compute_embedding_stats
from reasoning_agent.prompt import build_prompt
from reasoning_agent.report import Hypothesis, ReasoningReport

_DEFAULT_MODEL = "qwen3:8b"
_OLLAMA_HOST = "http://localhost:11434"


def _call_ollama(system: str, user: str, model: str) -> str:
    """Call Ollama chat API. Returns the model's text response."""
    try:
        import ollama
    except ImportError:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "temperature": 0.3,     # low temp for clinical reasoning
            "num_predict": 2048,
        },
        think=True,                 # Qwen3 extended thinking mode
    )
    return response.message.content


def _extract_json(text: str) -> str:
    """Extract the first JSON object from model output (strips <think> blocks etc.)."""
    # Strip <think>…</think> blocks produced by Qwen3 thinking mode
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text.strip(), flags=re.MULTILINE)
    # Find first { … } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _parse_response(raw: str) -> tuple[dict, Optional[str]]:
    """Parse the model response into a dict. Returns (data, error)."""
    try:
        cleaned = _extract_json(raw)
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        return {}, f"JSON parse error: {e}. Raw snippet: {raw[:300]}"


def analyze(
    feat: FeatureRecord,
    model: str = _DEFAULT_MODEL,
) -> ReasoningReport:
    """
    Run the full reasoning pipeline on a FeatureRecord.

    Parameters
    ----------
    feat  : populated FeatureRecord from feature_extractor.extract()
    model : Ollama model name (default: qwen3:8b)

    Returns
    -------
    ReasoningReport with hypotheses, flags, and recommendations.
    """
    # Compute embedding-based statistics
    emb_stats = {}
    if feat.lead_embeddings_available and feat.lead_embeddings is not None:
        emb_stats = compute_embedding_stats(feat.lead_embeddings, feat.channel_labels)

    # Build prompt
    system_prompt, user_message = build_prompt(feat, emb_stats)

    # Call model
    try:
        raw_response = _call_ollama(system_prompt, user_message, model)
    except Exception as e:
        return ReasoningReport(
            embedding_stats=emb_stats,
            model=model,
            error=f"Ollama call failed: {e}",
        )

    # Parse response
    data, parse_error = _parse_response(raw_response)
    if parse_error:
        return ReasoningReport(
            embedding_stats=emb_stats,
            raw_response=raw_response,
            model=model,
            error=parse_error,
        )

    # Build hypotheses
    hypotheses = []
    for h in data.get("hypotheses", []):
        try:
            hypotheses.append(Hypothesis(
                name=str(h.get("name", "")),
                description=str(h.get("description", "")),
                confidence=float(h.get("confidence", 0.0)),
                evidence_for=list(h.get("evidence_for", [])),
                evidence_against=list(h.get("evidence_against", [])),
            ))
        except (TypeError, ValueError):
            continue

    return ReasoningReport(
        hypotheses=hypotheses,
        most_likely=str(data.get("most_likely", "")),
        clinical_flags=list(data.get("clinical_flags", [])),
        recommended_next_steps=list(data.get("recommended_next_steps", [])),
        what_would_change_mind=list(data.get("what_would_change_mind", [])),
        caveats=list(data.get("caveats", [])),
        embedding_stats=emb_stats,
        raw_response=raw_response,
        model=model,
    )
