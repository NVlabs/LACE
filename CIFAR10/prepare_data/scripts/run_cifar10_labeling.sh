#!/usr/bin/env bash

# labeling
python cifar10_labeling.py -l -a densenet --dataset cifar10 --test-batch 20 \
--depth 190 --growthRate 40 --outdir ../ada-cifar10-latents --root_path ../ada-cifar10-latents \
--resume ../pretrained/classifiers/cifar10/densenet-bc-L190-k40/model_best.pth.tar
