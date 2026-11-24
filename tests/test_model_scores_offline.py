import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_model_scores_json_exists_and_valid():
    p = REPO / "results" / "model_scores.json"
    assert p.is_file(), "run scripts/render_model_scores.py first"
    doc = json.loads(p.read_text())
    assert len(doc["models"]) >= 10
    llama = next(m for m in doc["models"] if "Llama-3.1-8B" in m["model"])
    assert llama["grade"] == "S"
    assert llama["fast_eligible"] is True
    smol = next(m for m in doc["models"] if "SmolLM2-135M" in m["model"])
    assert smol["fast_eligible"] is False