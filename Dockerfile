# Seiden's own environment, from its pinned requirements (benchmarks/stanford/tasti/requirements.txt), for the R7.O3
# comparison. Seiden's code is mounted at run time (docker-compose.yml), not copied, so the image holds only its
# dependencies. The index runs on cached detector outputs, so a CPU build of PyTorch suffices.
FROM python:3.8-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# PyTorch of the same era as the pinned requirements, CPU wheels.
RUN pip install --no-cache-dir torch==1.7.1+cpu torchvision==0.8.2+cpu \
      -f https://download.pytorch.org/whl/torch_stable.html

# TASTI's pinned requirements (sklearn==0.0 is a deprecated alias of scikit-learn that PyPI no longer installs, so
# scikit-learn is pinned directly), plus the packages Seiden's own code imports. Seiden does not pin PyAV; 12.3.0 is the oldest release with Python
# 3.8 wheels still published, and it keeps the av.open / demux / is_keyframe API that src/iframes/pyav_utils.py uses.
RUN pip install --no-cache-dir \
      jsonlines==1.2.0 numba==0.50.1 pytorch-nlp==0.5.0 feather-format==0.4.1 scikit-learn==0.23.2 scipy==1.5.2 \
      tqdm==4.48.2 tabulate==0.8.7 \
      numpy==1.19.5 pandas==1.1.5 av==12.3.0 opencv-python-headless==4.5.1.48 Pillow==8.1.0

# Seiden's repository and its vendored benchmarks (TASTI, BlazeIt, SWAG, SUPG), mounted here by docker-compose.yml.
ENV PYTHONPATH=/nethome/jbang36/seiden:/nethome/jbang36/seiden/benchmarks/stanford/tasti:/nethome/jbang36/seiden/benchmarks/stanford/blazeit:/nethome/jbang36/seiden/benchmarks/stanford/swag_python:/nethome/jbang36/seiden/benchmarks/stanford/supg
WORKDIR /nethome/jbang36/seiden
