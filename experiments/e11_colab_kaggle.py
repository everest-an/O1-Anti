# =============================================================================
# E11 (MQAR: NLA vs attention vs Mamba) — GPU runner for Colab / Kaggle.
#
# Mamba's CPU sequential fallback is ~6.7 s/step (11 h per 6000-step cell) — the
# whole sweep is ~66 h on CPU, infeasible. On a GPU with the mamba-ssm /
# causal-conv1d CUDA kernels it is minutes. This runs the full decisive sweep.
#
# HOW TO USE
#   Colab : Runtime -> Change runtime type -> GPU (T4 is fine). Paste this whole
#           file into one cell and run.
#   Kaggle: New Notebook -> Settings -> Accelerator -> GPU T4 x1. Paste and run.
#           (Enable "Internet" in Kaggle settings so pip/clone work.)
#
# Paste the printed result table (the block starting "pairs   attention ...")
# back into the chat.
# =============================================================================
import os, subprocess, sys

REPO = "https://github.com/everest-an/O1-Anti.git"
ROOT = "/kaggle/working" if os.path.isdir("/kaggle") else "/content"
DEST = os.path.join(ROOT, "O1-Anti")


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=False)


# --- 1. get the code (fresh clone or pull latest) -------------------------
if not os.path.isdir(DEST):
    sh(f"git clone --depth 1 {REPO} {DEST}")
else:
    sh(f"cd {DEST} && git pull --ff-only")

# --- 2. deps. torch is preinstalled on Colab/Kaggle GPU images. -----------
#     transformers provides Mamba; the two kernel wheels give the fast path
#     (script still runs, slower, if a wheel fails to build — that's fine on GPU).
sh(f"{sys.executable} -m pip -q install 'transformers>=4.40' numpy")
sh(f"{sys.executable} -m pip -q install causal-conv1d>=1.2.0 || true")
sh(f"{sys.executable} -m pip -q install mamba-ssm>=2.0.0 || true")

# --- 3. sanity: is a GPU actually visible? --------------------------------
import torch
print(f"\ntorch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (SLOW — enable GPU!)'}\n",
      flush=True)

# --- 4. run the full decisive sweep ---------------------------------------
#   6000 steps = the budget that reliably groks these threshold-convergence
#   retrieval tasks (learned from E10); 2 seeds for an honest mean±std; pairs
#   8/16/32 to stress recall capacity where an SSM's fixed state should break.
cmd = (
    f"cd {DEST} && HF_HUB_OFFLINE=1 {sys.executable} -u experiments/e11_mqar_vs_ssm.py "
    f"--steps 6000 --pairs 8 16 32 --queries 8 --seeds 0 1 "
    f"--archs attention nla mamba --batch 32 --d_model 96 --device cuda"
)
sh(cmd)

print("\n\n=== DONE. Copy the 'pairs / attention / nla / mamba' table above back "
      "into the chat. ===", flush=True)
