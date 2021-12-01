#!/usr/bin/env bash

# annotate generated images with clip model
# Note: first install clip via `pip install git+https://github.com/openai/CLIP.git`
for threshold in 0.3 0.4 0.5 0.6; do

  python afhq_clip_labeling.py --threshold $threshold

done
