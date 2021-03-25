"""ML task runner."""

from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.contrib import training

from trainer import input_utils
from trainer import model


def _parse_arguments(argv):
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job_dir',
        help='Output directory for writing checkpoints and export models',
        required=True)
    parser.add_argument(
        '--train_file',
        help='Tfrecord file with training records',
        required=True)
    parser.add_argument(
        '--eval_file',
        help='Tfrecord file with eval records',
        required=True)
    parser.add_argument(
        '--batch_size',
        help='Batch size for each train step',
        type=int,
        default=12)
    parser.add_argument(
        '--num_epochs',
        help='Maximum number of iterations through the dataset',
        type=int,
        default=100)
    parser.add_argument(
        '--train_steps',
        help='Maximum number of training steps',
        type=int,
        default=10000)
    parser.add_argument(
        '--learning_rate',
        help='Learning rate',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--dropout_rate',
        help='Dropout rate',
        type=float,
        default=0.1)

    return parser.parse_args(argv)

def run_experiment(hparams):
    """Train and evaluate tf.estimator model"""

    train_spec = tf.estimator.TrainSpec(input_fn=input_utils._get_train_input_fn(hparams.train_file,
                                                                          hparams.batch_size,
                                                                          hparams.num_epochs,
                                                                          tf.estimator.ModeKeys.TRAIN),
                                        max_steps=hparams.train_steps)
    final_exporter = tf.estimator.FinalExporter('final_exporter', input_utils.serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_utils._get_train_input_fn(hparams.eval_file,
                                                                        hparams.batch_size,
                                                                        hparams.num_epochs,
                                                                        tf.estimator.ModeKeys.EVAL),
                                      exporters=[final_exporter])
    # Checkpoints to save
    run_config = tf.estimator.RunConfig(model_dir=hparams.job_dir, save_checkpoints_steps=100, keep_checkpoint_max=200)
    estimator = model.create_estimator(run_config, hparams)
    return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main():
    args = _parse_arguments(sys.argv[1:])
    hparams = training.HParams(**args.__dict__)
    tf.logging.set_verbosity(tf.logging.INFO)
    run_experiment(hparams)


if __name__ == '__main__':
  main()