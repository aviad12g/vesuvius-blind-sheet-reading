#!/bin/bash
# Scout: blind sweep (every tracked sheet, Reader V3 on both faces) over base meshes of five eligible scrolls never swept before.
R=/workspace/s3; cd $R; O=$R/o_scout; mkdir -p $O; exec > >(tee -a $O/run.log) 2>&1
t0=$(date +%s); el(){ echo $(( $(date +%s) - t0 )); }; log(){ echo "[SCOUT $(el)s] $*"; }
LIMIT=${LIMIT:-9000}        # seconds after which no new mesh is started
run(){ s=$1; shift
  for m in "$@"; do
    if [ -f $O/$s/$m/done.json ]; then continue; fi
    if [ $(el) -gt $LIMIT ]; then log "time limit reached, skipping $s $m"; continue; fi
    log "start $s $m"
    python3 sweep.py --scroll $s --only $m --D 170 --tile 1024 --marg 64 --ds 4 --models v3 --trace-faces forward,reverse --max-minutes 25 --out $O/$s 2>&1 | grep -a -E "t1447|canvases|READER_FAILED|Traceback|Error|error:|skipped" | cut -c1-400
    log "done $s $m"
  done
  (cd $O && tar czf /workspace/scout_$s.tgz --exclude='canv.npz' $s 2>/dev/null); ls -la /workspace/scout_$s.tgz | awk '{print $5, $9}'; log "PACKED $s"
}
run PHerc0268 z12272_w060 z3872_w060 z12272_w040 z3872_w040 z12272_w020 z3872_w020
run PHerc0257 z3744_w060 z4352_w060 z3744_w040 z4352_w040 z3744_w020 z4352_w020
run PHerc0358 z11280_w060 z11280_w040 z11280_w020
run PHerc0826 z6528_w060 z6528_w040 z6528_w020
run PHerc0125 z9936_w060 z9936_w040 z9936_w080 z6336_w060 z14144_w060 z17136_w040
cp $O/run.log /workspace/scout_run.log; log SCOUT_ALL_DONE
