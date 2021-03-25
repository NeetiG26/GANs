"""Input functions"""
from __future__ import print_function

import os
import multiprocessing

import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()


tag = ['blonde hair','brown hair','black hair','blue hair','pink hair',
               'purple hair','green hair','red hair','silver hair','white hair','orange hair',
               'aqua hair','gray hair','long hair','short hair','twintails','drill hair','ponytail','blush',
               'smile','open mouth','hat','ribbon','glasses','blue eyes','red eyes','brown eyes',
               'green eyes','purple eyes','yellow eyes','pink eyes','aqua eyes','black eyes','orange eyes',]
tag_map = dict()
for i, j in enumerate(tag):
    tag_map[j] = i


def one_hot_fn(x):
    one_hot = np.zeros(len(tag))
    vals = []
    for ele in x:
        vals.append(tf.compat.as_text(ele))
    one_hot[list(map(lambda each: tag_map[each], vals))] = 1
    one_hot = np.array(one_hot, dtype=np.float32)
    return one_hot

def preprocess_(x):
    x = tf.strings.split(x,sep=',')
    x = x.values
    features = x[-2]
    features = tf.strings.split([features],sep=';')
    features = features.values
    features = tf.py_func(one_hot_fn, [features], tf.float32)
    path = x[-1]
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    #the shape should be 128, 128
    image = tf.image.resize(image, [128, 128 ])
    return image, features


def _get_train_input_fn(file_dir, batch_size=64, noise_dims=100, dataset_dir=None,
                        num_threads=4, shuffle_buffer_size=2):
    def train_input_fn():
        data = tf.data.TextLineDataset(file_dir)
        data = data.map(lambda *x : preprocess_(x))
        data = data.batch(batch_size, drop_remainder=False)
        data = data.repeat()
        data = data.shuffle(buffer_size=shuffle_buffer_size)
        data = data.prefetch(buffer_size=2*batch_size)
        iterator = data.make_one_shot_iterator()
        noise = tf.random_normal([batch_size, 128])
        image, features = iterator.get_next()
        return {'image':image, 'noise': noise, 'features':features}, image
    return train_input_fn

# def serving_input_fn():

#     """Builds ServingInputReceiver to convert placeholders to features."""
#     example = tf.placeholder(dtype=tf.string,
#                                          shape=[None, 64, 64, 3],
#                                          name='input_example_tensor')
#     receiver_tensors = {'examples': example}
#     keys_to_features = {
#         'noise' : tf.VarLenFeature(tf.float32)
#         }
#     features = tf.parse_example(example, keys_to_features)
#     return tf.estimator.export.ServingInputReceiver(features,receiver_tensors)
