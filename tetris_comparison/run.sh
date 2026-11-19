#!/bin/sh
# Run from the local experiment workspace: Seiden on ace in its own container, scored in ace's polyis container.
set -eu
checkout=$(cd "$(dirname "$0")/.." && pwd)
workspace=$(cd "$checkout/.." && pwd)
remote=/work/cwkt/projects/seiden
data=/data/cwkt/data/polyis-cache/seiden

if [ "$#" -eq 0 ]; then
    set -- caldot1-y05 caldot2-y05 ams-y05 jnc0 jnc2 jnc6 jnc7
fi

rsync -a --exclude=.git --exclude=__pycache__ "$checkout/" "ace:$remote/"
ssh -o BatchMode=yes ace bash -s -- "$@" <<REMOTE
set -eu
cd $remote
mkdir -p $data/logs $data/torch_cache
for dataset do
    # Seiden's inputs from Tetris's saved reference-detector outputs, built once.
    [ -f $data/video_data/\$dataset/manifest.json ] ||
        docker exec -w /polyis -e PYTHONPATH=/polyis polyis python measurements/r7o3_seiden_prepare.py --datasets \$dataset
    rm -rf $data/tasti_data/results/\$dataset $data/tasti_data/cache/\$dataset
    # One container per seed.
    for seed in 0 1 2; do
        SEIDEN_DATA=$data docker compose run --rm -T seiden --dataset \$dataset --seeds \$seed \
            > $data/logs/\${dataset}_s\$seed.log 2>&1 &
    done
done
wait
for dataset do
    rm -rf /work/cwkt/projects/tetris/measurements-results/r7o3_seiden/\$dataset
    mkdir -p /work/cwkt/projects/tetris/measurements-results/r7o3_seiden
    cp -r $data/tasti_data/results/\$dataset /work/cwkt/projects/tetris/measurements-results/r7o3_seiden/
done
docker exec -w /polyis -e PYTHONPATH=/polyis polyis python measurements/r7o3_seiden_score.py --datasets "\$@" > /dev/null
REMOTE
for dataset do
    rm -rf "$workspace/tetris/measurements-results/r7o3_seiden/$dataset"
done
rsync -a ace:/work/cwkt/projects/tetris/measurements-results/r7o3_seiden/ "$workspace/tetris/measurements-results/r7o3_seiden/"
rsync -a ace:/work/cwkt/projects/tetris/measurements-results/r7o3_seiden_score/ "$workspace/tetris/measurements-results/r7o3_seiden_score/"
