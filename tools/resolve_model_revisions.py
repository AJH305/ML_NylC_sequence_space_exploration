"""Record immutable ESM-2 revisions without downloading model weights."""

import json
from pathlib import Path
import urllib.request

models = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t33_650M_UR50D"]
result = {}
for model in models:
    with urllib.request.urlopen("https://huggingface.co/api/models/" + model) as response:
        info = json.load(response)
    result[model] = {"revision": info["sha"], "source": "https://huggingface.co/" + model}
path = Path(__file__).resolve().parents[1] / "references/model_revisions.json"
path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result))
