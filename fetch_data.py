"""Fetch the published data and generate the native Harbor task directories."""
import json
import shutil
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
snapshot = Path(snapshot_download("mohammed8284/inkling-underwriting-eval", repo_type="dataset",
                revision="32eaacb180f18351128f91d3e35f22fb139f0364", token=False))
for name in ("cases.json", "corpus.json", "source-scope.json"):
    shutil.copyfile(snapshot / name, root / "benchmark" / name)
shutil.copytree(snapshot / "results", root / "results", dirs_exist_ok=True)
sys.path.insert(0, str(root / "benchmark"))
import build
build.build_harbor(build.load_cases(), json.loads((snapshot / "corpus.json").read_text()))
print("Downloaded data and results. Generated 15 native Harbor tasks in benchmark/harbor/.")
