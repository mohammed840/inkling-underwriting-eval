from pathlib import Path
import subprocess
import uuid
import modal

root = Path(__file__).resolve().parent
app = modal.App("inkling-underwriting-eval")
volume = modal.Volume.from_name("inkling-underwriting-trials", create_if_missing=True)
cookbook = "git+https://github.com/thinking-machines-lab/tinker-cookbook.git@1991897cba59ea9d9b0030d183d1867b28eba9d9"
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
         .pip_install("torch==2.15.0", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install_from_requirements(str(root / "requirements.txt"))
         .run_commands(f"pip install --no-deps {cookbook}")
         .add_local_dir(root / "benchmark", "/opt/eval/benchmark", ignore=["harbor/**","**/__pycache__/**"])
         .add_local_file(root / "evaluate.py", "/opt/eval/evaluate.py")
         .add_local_file(root / "pricing.json", "/opt/eval/pricing.json"))

@app.function(image=image, volumes={"/trials": volume}, timeout=43200, retries=0,
              secrets=[modal.Secret.from_local_environ(["OPENAI_API_KEY","TINKER_API_KEY"])])
def run(model: str, trial: str):
    try:
        subprocess.run(["python", "/opt/eval/evaluate.py", f"/trials/{trial}", "--execute", "--model", model], check=True)
    finally:
        volume.commit()

@app.local_entrypoint()
def main(model: str = "Inkling"):
    trial = uuid.uuid4().hex
    print("Volume trial directory:", trial, flush=True)
    run.remote(model, trial)
