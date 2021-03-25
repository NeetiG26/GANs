#! /usr/bin/env bash

# Arguments passed to the module (i.e. task.py)
# shellcheck disable=SC2154
export MODULE_ARGS=" \
  --train_file="gs://dimensionality_reduction/dummy_data/train.tfrecord"\
  --eval_file="gs://dimensionality_reduction/dummy_data/test.tfrecord"\
  --shape=500\
  --batch_size=32\
  --num_epochs=100\
  --train_steps=10000\
  --learning_rate=0.001\
  --l2_reg=0.0001\
  --noise_level=0.0\
  --dropout_rate=0.1\
  "
