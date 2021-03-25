"""Input functions"""
from __future__ import print_function

import os
import multiprocessing

import tensorflow as tf


def preprocess_image(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [64, 64])
    print(image)
    image = image-127.5/127.5
    return image


def _get_train_input_fn(file_dir, batch_size=32, noise_dims=100, dataset_dir=None,
                        num_threads=4, shuffle_buffer_size=2):
    def train_input_fn():
        path_tf = tf.data.Dataset.list_files(os.path.expanduser(file_dir))
        image_tf = path_tf.map(preprocess_image)
        image_tf = image_tf.batch(batch_size, drop_remainder=False)
        image_tf = image_tf.repeat()
        image_tf = image_tf.shuffle(buffer_size=shuffle_buffer_size)
        image_tf = image_tf.prefetch(buffer_size=2*batch_size)
        iterator = image_tf.make_one_shot_iterator()
        noise = tf.random_normal([batch_size, noise_dims])
        image = iterator.get_next()
        return {'image':image}, 0
    return train_input_fn

def serving_input_fn():

    """Builds ServingInputReceiver to convert placeholders to features."""
    example = tf.placeholder(dtype=tf.string,
                                         shape=[None, 64, 64, 3],
                                         name='input_example_tensor')
    receiver_tensors = {'examples': example}
    keys_to_features = {
        'noise' : tf.VarLenFeature(tf.float32)
        }
    features = tf.parse_example(example, keys_to_features)
    return tf.estimator.export.ServingInputReceiver(features,receiver_tensors)
