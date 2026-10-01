#!/usr/bin/env bash
#SBATCH --job-name=hippo-arm-tf-build
#SBATCH --account=brics.u6tj
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
C=${1:?deployment root}
export APPTAINER_CACHEDIR="$C/container-cache"
export APPTAINER_TMPDIR="$C/container-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$C/containers" "$C/env-osgpr/bin"
IMAGE="$C/containers/tensorflow-24.02-tf2-py3.sif"
if [[ ! -f "$IMAGE" ]]; then
  apptainer pull "$IMAGE" docker://nvcr.io/nvidia/tensorflow:24.02-tf2-py3
fi
apptainer exec --bind "$C:$C" "$IMAGE" python3 -m venv --system-site-packages "$C/env-osgpr-real"
apptainer exec --bind "$C:$C" "$IMAGE" "$C/env-osgpr-real/bin/python" -m pip install --no-deps \
  gpflow==2.9.0 tensorflow-probability==0.23.0 check-shapes==1.1.1 deprecated==1.2.14 \
  multipledispatch==1.0.0 tabulate==0.9.0 dropstackframe==0.1.1
# Argument arrays and the binding preserve the selected DoC artifact paths.
cat > "$C/env-osgpr/bin/python" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec apptainer exec --nv --bind "$C:$C,$C:/vol/bitbucket/nk523/hipposvgp-fair-20260929" "$IMAGE" "$C/env-osgpr-real/bin/python" "\$@"
EOF
chmod 700 "$C/env-osgpr/bin/python"
apptainer exec --bind "$C:$C" "$IMAGE" "$C/env-osgpr-real/bin/python" -m pip freeze > "$C/environment-records/env-osgpr.txt"
sha256sum "$IMAGE" > "$C/environment-records/tensorflow-container.sha256"
