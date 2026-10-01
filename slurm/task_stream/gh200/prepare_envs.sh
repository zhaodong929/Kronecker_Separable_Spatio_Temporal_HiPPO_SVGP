#!/usr/bin/env bash
# Isolated ARM qualification environments. Never modifies an existing environment.
set -euo pipefail
C=${1:?deployment root}
UV=${UV:-/home/u6tj/nk523.u6tj/.local/bin/uv}
BASE=/scratch/u6tj/nk523.u6tj/venvs/interdomain-e364477-cu126
mkdir -p "$C/environment-records"
for name in env-routeb env-jax-gpu env-tracking; do
  if [[ ! -e "$C/$name/bin/python" ]]; then
    "$UV" venv --python "$BASE/bin/python" "$C/$name"
  fi
done
"$UV" pip install --python "$C/env-routeb/bin/python" numpy==1.26.4 scipy==1.12.0 matplotlib pandas pytest
# Reuse the installed ARM CUDA Torch distribution read-only; local packages
# precede it on sys.path. Record the inherited package inventory as well.
"$C/env-routeb/bin/python" - "$BASE" <<'PY'
import pathlib,sys,sysconfig
base=pathlib.Path(sys.argv[1])/'lib/python3.11/site-packages'
assert (base/'torch').is_dir()
(pathlib.Path(sysconfig.get_paths()['purelib'])/'gh200-torch-base.pth').write_text(str(base)+'\n')
PY
"$UV" pip install --python "$C/env-tracking/bin/python" numpy==1.26.4 wandb==0.25.1
"$UV" pip install --python "$C/env-jax-gpu/bin/python" 'jax[cuda12]==0.4.35' objax==1.8.0 bayesnewton==1.1 numpy==1.26.4 scipy==1.12.0 matplotlib pandas pytest
for name in env-routeb env-jax-gpu env-tracking; do
  "$UV" pip freeze --python "$C/$name/bin/python" > "$C/environment-records/$name.txt"
done
"$UV" pip freeze --python "$BASE/bin/python" > "$C/environment-records/inherited-torch-base.txt"
