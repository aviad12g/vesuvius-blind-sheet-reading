#!/bin/bash
# all-sheets sweep (9.362 um scrolls) — pod setup. Public, pinned sources; weights sha256-checked.
set -euo pipefail
R=${R:-/workspace/s3}; mkdir -p $R/weights; cd $R
t0=$(date +%s); log(){ echo "[$(( $(date +%s) - t0 ))s] $*"; }
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
log pip
pip install -q "zarr>=3" numcodecs tifffile timm einops fsspec aiohttp huggingface_hub opencv-python-headless \
  scipy safetensors requests pynrrd imagecodecs pillow pyyaml tqdm 2>&1 | grep -v -i "warning" | tail -3 || true
sparse(){ # dir url commit paths...
  local d=$1 u=$2 c=$3; shift 3; rm -rf $d; git init -q $d; cd $d; git remote add origin $u
  git sparse-checkout init --cone; git sparse-checkout set "$@"
  git fetch -q --depth 1 --filter=blob:none origin $c; git checkout -q FETCH_HEAD; echo "$d $(git rev-parse HEAD)"; cd $R; }
log "306B first-letters repo (code + winning mesh)"
sparse r https://github.com/SirProdigle/pherc0306b-first-letters.git main submission/code submission/mesh
log "villa @ a329895e (vesuvius/src)"
sparse villa https://github.com/ScrollPrize/villa.git a329895ea5e9bde78752e60dbc0a4878dd2ab909 vesuvius/src
log "Studio ink model code @ 235e3be8 (PHerc0211 winner)"
sparse studio https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt.git 235e3be8a06749344862c89a1e52c81d90b15e0e studio-multiscroll-cpt-20261023/hf
log "public meshes @ 620769e2 (PHerc0813, PHerc0358, PHerc0211)"
sparse pmesh https://github.com/pscamillo/vesuvius-eligible-meshes.git 620769e2e1e70d1e61b588092229cb76d3af4804 meshes/PHerc0813 meshes/PHerc0358 meshes/PHerc0211
log weights
C=$R/r/submission/code; mkdir -p $C/models/community/trace
curl -sfL --retry 5 -o $C/models/community/trace/v3B_best.pth https://github.com/Jashann/vesuvius-scrolling/releases/download/v1.0/v3B_best.pth
curl -sfL --retry 5 -o $R/weights/studio-softloss-20261023-ema.safetensors \
  https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/releases/download/studio-softloss-20261023/studio-softloss-20261023-ema.safetensors
mkdir -p $C/models/community/reader-v3
curl -sfL --retry 5 -o $C/models/community/reader-v3/reader-v3.pth https://huggingface.co/domenicor046/reader-v3/resolve/main/reader-v3.pth
curl -sfL --retry 5 -o $C/third_party/v8in/model.safetensors https://huggingface.co/YoussefMoNader/ink-8um-v8in/resolve/main/model.safetensors
sha256sum -c <<EOF
812ffe2b42b3293974052ebbee60a0fe939ca2bfc3f03a517a0bc2e2339e5da2  $C/models/community/trace/v3B_best.pth
96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90  $R/weights/studio-softloss-20261023-ema.safetensors
ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a  $C/models/community/reader-v3/reader-v3.pth
3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd  $C/third_party/v8in/model.safetensors
EOF
for s in PHerc0813 PHerc0358 PHerc0211; do echo $s $(ls $R/pmesh/meshes/$s | wc -l); done; ls $R/r/submission/mesh
log SETUP_OK
