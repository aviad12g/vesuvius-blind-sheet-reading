#!/bin/bash
# PHerc1447 control: the team's surface, our blind sweep and our automatic re-mesh, scored against the public ink labels.
R=/workspace/s3; cd $R; O=$R/o1447; mkdir -p $O; exec > >(tee -a $O/run.log) 2>&1
t0=$(date +%s); log(){ echo "[RUN $(( $(date +%s) - t0 ))s] $*"; }
U=https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/main
if [ ! -f $O/DATA_OK ]; then
  for w in w058 w060 w062; do mkdir -p hf/$w/tifxyz hf/$w/labels hf/$w/predictions
    for f in tifxyz/x.tif tifxyz/y.tif tifxyz/z.tif predictions/v8in.png; do curl -sfL --retry 5 -o hf/$w/$f $U/$w/$f; done; done
  for w in w058 w060; do for f in labels/inklabels.png labels/mask.png; do curl -sfL --retry 5 -o hf/$w/$f $U/$w/$f; done; done
  curl -sfL --retry 5 -o hf/SHA256SUMS $U/SHA256SUMS
  (cd hf && grep -E "  (w058|w060|w062)/(tifxyz/[xyz]\.tif|labels/.*|predictions/v8in\.png)$" SHA256SUMS | sha256sum -c --quiet) || { log "HF checksum FAILED"; exit 1; }
  python3 mk1447.py hf wm1447 || exit 1
  grep -q PHerc1447 sweep.py || sed -i "s/^SCROLLS = {/SCROLLS = {'PHerc1447': _sc('PHerc1447', '20250521151220', 8.64, 116), /" sweep.py
  grep -c PHerc1447 sweep.py; touch $O/DATA_OK; log DATA_OK
fi
COMMON="--scroll PHerc1447 --tile 1024 --marg 64 --ds 4 --max-minutes 80"
if [ ! -f $O/A_OK ]; then
  for b in a b; do log "A band $b: team surface through our code"
    python3 sweep.py $COMMON --meshdir $R/wm1447 --only t1447_w058$b --D 24 --base-only --models v3,v8,studio --studio-stride 96 --studio-faces forward,reverse --trace-faces forward,reverse --out $O/A_$b; done
  touch $O/A_OK; log A_OK
fi
if [ ! -f $O/B_OK ]; then
  for b in a b; do log "B band $b: blind tracking of every sheet, Reader V3 on both faces"
    python3 sweep.py $COMMON --meshdir $R/wm1447 --only t1447_w058$b --D 170 --models v3 --trace-faces forward,reverse --out $O/B_$b; done
  touch $O/B_OK; log B_OK
fi
python3 score1447.py $O hf wm1447 > $O/score1.txt 2>&1; tail -3 $O/score1.txt; log SCORE1_DONE
if [ "$1" = "stop1" ]; then log STOP1; exit 0; fi
if [ ! -f $O/C_OK ]; then
  FACE=$(python3 -c "import json; print(json.load(open('$O/sel.json'))['face']['v8'])")
  for b in a b; do CV=$(python3 -c "import json; print(','.join(str(c) for c in json.load(open('$O/sel.json'))['canv']['$b']))")
    if [ -n "$CV" ]; then log "C band $b: v8-in ($FACE) on canvases $CV"
      python3 sweep.py $COMMON --meshdir $R/wm1447 --only t1447_w058$b --D 170 --models v8 --trace-faces $FACE --canv $CV --out $O/C_$b; fi; done
  touch $O/C_OK; log C_OK
fi
if [ ! -f $O/R_OK ]; then
  CI=$(python3 -c "import json; r=json.load(open('$O/sel.json'))['remesh']; print(r['canvas'] if r else '')")
  if [ -n "$CI" ]; then log "R: re-mesh canvas $CI of band b and read it"
    python3 tools9.py remesh $R/wm1447/t1447_w058b $O/B_b/t1447_w058b/canv.npz $CI $R/wm1447r/r1_w058 1058 1892 750 1880 10 0 && \
    python3 meshcheck.py $R/wm1447r/r1_w058 $O/fig_meshcheck_r1.png > $O/meshcheck_r1.json 2>&1
    python3 sweep.py $COMMON --meshdir $R/wm1447r --only r1_w058 --D 24 --base-only --models v3,v8 --trace-faces forward,reverse --out $O/R; fi
  touch $O/R_OK; log R_OK
fi
python3 score1447.py $O hf wm1447 $R/wm1447r/r1_w058 $O/R > $O/score2.txt 2>&1; tail -3 $O/score2.txt
cd $R && tar czf res1447.tgz o1447/run.log o1447/score1.txt o1447/score2.txt o1447/score.json o1447/sel.json o1447/fig o1447/meshcheck_r1.json o1447/fig_meshcheck_r1.png o1447/*/*/done.json 2>/dev/null; ls -la res1447.tgz
log ALL_DONE
