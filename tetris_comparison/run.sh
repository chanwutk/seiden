#!/bin/sh
# Run from the local experiment workspace: Seiden on ace in its own container, scored in ace's polyis container.
set -eu
checkout=$(cd "$(dirname "$0")/.." && pwd)
workspace=$(cd "$checkout/.." && pwd)
remote=/work/cwkt/projects/seiden
tetris=/work/cwkt/projects/tetris
data=/data/cwkt/data/polyis-cache/seiden

if [ "$#" -eq 0 ]; then
    set -- caldot1-y05 caldot2-y05 ams-y05 jnc0 jnc2 jnc6 jnc7
fi

# This repository, and Tetris's input preparation and scoring scripts.
rsync -a --exclude=.git --exclude=__pycache__ "$checkout/" "ace:$remote/"
rsync -a "$workspace/tetris/measurements/r7o3_seiden_prepare.py" "$workspace/tetris/measurements/r7o3_seiden_score.py" \
    "ace:$tetris/measurements/"
ssh -o BatchMode=yes ace bash -s -- "$@" <<REMOTE
set -eu
cd $remote
mkdir -p $data/logs $data/torch_cache
SEIDEN_DATA=$data docker compose build -q
# Per-clip inputs from Tetris's saved reference-detector outputs; the prepared datasets must include the tests' inputs.
docker exec -w /polyis -e PYTHONPATH=/polyis polyis python measurements/r7o3_seiden_prepare.py \
    --datasets \$(printf '%s\n' "\$@" jnc0 caldot1-y05 | sort -u)
# The wrappers must leave Seiden's answers unchanged before any measured run (stdin closed so the container cannot
# consume the rest of this script, which bash reads from stdin).
SEIDEN_DATA=$data docker compose run --rm -T --entrypoint python seiden \
    -m pytest -p no:cacheprovider tetris_comparison/test_run.py -q < /dev/null
# One container per dataset; stop if any dataset's run fails.
pids=
for dataset do
    rm -rf $data/tasti_data/results/\$dataset $data/tasti_data/cache/\$dataset
    SEIDEN_DATA=$data docker compose run --rm -T seiden --dataset \$dataset < /dev/null > $data/logs/\$dataset.log 2>&1 &
    pids="\$pids \$!"
done
status=0
for pid in \$pids; do
    wait \$pid || status=1
done
[ \$status -eq 0 ] || { echo "Seiden failed; see $data/logs" >&2; exit 1; }
for dataset do
    rm -rf $tetris/measurements-results/r7o3_seiden/\$dataset
    mkdir -p $tetris/measurements-results/r7o3_seiden
    cp -r $data/tasti_data/results/\$dataset $tetris/measurements-results/r7o3_seiden/
done
docker exec -w /polyis -e PYTHONPATH=/polyis polyis python measurements/r7o3_seiden_score.py --datasets "\$@" > /dev/null
echo "Seiden runs scored for: \$*"
REMOTE
for dataset do
    rm -rf "$workspace/tetris/measurements-results/r7o3_seiden/$dataset"
done
rsync -a ace:$tetris/measurements-results/r7o3_seiden/ "$workspace/tetris/measurements-results/r7o3_seiden/"
rsync -a ace:$tetris/measurements-results/r7o3_seiden_score/ "$workspace/tetris/measurements-results/r7o3_seiden_score/"
# Each dataset must come back with this run's results (one file per query and budget fraction).
for dataset do
    count=$(find "$workspace/tetris/measurements-results/r7o3_seiden/$dataset" -name '*_f*_s*.json' | wc -l)
    [ "$count" -eq 16 ] || { echo "$dataset: expected 16 Seiden runs, found $count" >&2; exit 1; }
done
