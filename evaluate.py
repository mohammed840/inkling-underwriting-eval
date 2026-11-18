"""Prepare or execute a fresh pass@4 run with the exported benchmark."""
import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmark"))
import run_luna_eval as evaluator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="New run directory")
parser.add_argument("--model", choices=["Inkling", "Inkling-Small", "Inkling-2-Small"], default="Inkling")
parser.add_argument("--execute", action="store_true", help="Make paid candidate and judge calls")
args = parser.parse_args()
evaluator.TOKENIZER = "thinkingmachines/" + args.model
evaluator.CANDIDATE = evaluator.TOKENIZER + ":peft:262144"
evaluator.BUDGET = 40.0 if args.model == "Inkling" else 28.0
args.output.mkdir(parents=True, exist_ok=False)
ledger = args.output / "prior"
ledger.mkdir()
(ledger / "initial.json").write_text(json.dumps({"reservation_usd": 0}))
shutil.copy2(ROOT / "pricing.json", args.output)
(args.output / "PROTOCOL.md").write_text(
    f"{args.model}, four independent three-turn attempts per case, seeds 20261117–20261120, "
    f"temperature 0.6. Direct OpenAI Luna. All turns and criteria required. "
    f"Fresh run ledger, frozen rates, reservation cap USD {evaluator.BUDGET}.\n"
)
asyncio.run(evaluator.run(args.output, args.execute, prior_ledger=ledger, attempts=4))
