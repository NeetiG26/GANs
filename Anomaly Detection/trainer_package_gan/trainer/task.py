import tensorflow as tf
import os
# from trainer import metrics
import tensorflow.layers as layers
import tensorflow.contrib.gan as tfgan
from tensorflow.contrib.gan.python.namedtuples import GANTrainSteps
import argparse


def get_generator_fn(batch_size):
    def generator_fn(input_image, mode):
        with tf.name_scope('generator'):
            #4*4
            dense_1 = layers.dense(inputs=input_image, units=batch_size*16)
            batch_norm_1 = layers.batch_normalization(inputs=dense_1)
            reshape_1 = tf.reshape(batch_norm_1, shape=(batch_size, 4, 4, batch_size))
            relu_1 = tf.nn.relu(reshape_1)
            # 8*8
            conv_T_1 = layers.conv2d_transpose(inputs=relu_1, filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')
            batch_norm_2 = layers.batch_normalization(inputs=conv_T_1)
            relu_2 = tf.nn.relu(batch_norm_2)
            # 16*16
            conv_T_2 = layers.conv2d_transpose(inputs=relu_2, filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')
            batch_norm_3 = layers.batch_normalization(inputs=conv_T_2)
            relu_3 = tf.nn.relu(batch_norm_3)
            # 32*32
            conv_T_3 = layers.conv2d_transpose(inputs=relu_3, filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')
            batch_norm_4 = layers.batch_normalization(inputs=conv_T_3)
            relu_4 = tf.nn.relu(batch_norm_4)
            # 64*64
            conv_T_4 = layers.conv2d_transpose(
                inputs=relu_4, filters=3, kernel_size=(2, 2), strides=(2, 2), padding='same')
            tanh_1 = tf.nn.tanh(conv_T_4)
            print(tanh_1)
            return tanh_1
    return generator_fn


def discriminator_fn(image, noise):
    with tf.name_scope('Discriminator'):
        # 64 -> 32
        conv_1 = layers.conv2d(image, 64 , (2, 2), padding='same')
        lrelu_1 = tf.nn.leaky_relu(conv_1, alpha=0.2)
        # 32 -> 16
        conv_2 = layers.conv2d(lrelu_1, 64, (2, 2), padding='same')
        batch_norm_1 = layers.batch_normalization(inputs=conv_2)
        lrelu_2 = tf.nn.leaky_relu(batch_norm_1, alpha=0.2)
        # 16 -> 8
        conv_3 = layers.conv2d(lrelu_2, 64, (2, 2), padding='same')
        batch_norm_2 = layers.batch_normalization(inputs=conv_3)
        lrelu_3 = tf.nn.leaky_relu(batch_norm_2, alpha=0.2)
        # 8 -> 4
        conv_4 = layers.conv2d(lrelu_3, 64, (2,2), padding='same')
        batch_norm_3 = layers.batch_normalization(inputs=conv_4)
        lrelu_4 = tf.nn.leaky_relu(batch_norm_3, alpha=0.2)
        fc1 = layers.flatten(lrelu_4)
        fc2 = layers.dense(fc1, 1)
        return fc2

def serving_input_fn():

    """Builds ServingInputReceiver to convert placeholders to features."""
    example = tf.placeholder(dtype=tf.string,
                                         shape=[None, 100],
                                         name='input_example_tensor')
    receiver_tensors = {'examples': example}
    keys_to_features = {
        'noise' : tf.VarLenFeature(tf.float32)
        }
    features = tf.parse_example(example, keys_to_features)
    return tf.estimator.export.ServingInputReceiver(features,receiver_tensors)

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
        return noise, image
    return train_input_fn


def _get_predict_input_fn(batch_size, noise_dims):
    def predict_input_fn():
        noise = tf.random_normal([batch_size, noise_dims])
        return noise
    return predict_input_fn


def get_run_config(check_point_dir, summary_steps, checkpoints_steps):
    run_config = tf.estimator.RunConfig(
        model_dir=check_point_dir,
        save_summary_steps=summary_steps,
        save_checkpoints_steps=checkpoints_steps)
    return run_config


def end_to_end(check_point_dir, batch_size, image_path, summary_steps=100, checkpoints_steps=100, max_training_steps=10000, eval_steps=5000):
    
    train_input_fn = _get_train_input_fn(image_path, batch_size)
    eval_input_fn = _get_train_input_fn(eval_image_path, batch_size)
    gan_estimator = tfgan.estimator.GANEstimator(
        model_dir=check_point_dir,
        generator_fn=get_generator_fn(batch_size),
        discriminator_fn=discriminator_fn,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.00003, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.00001, 0.5),
        get_hooks_fn= tf.contrib.gan.get_sequential_train_hooks(train_steps=GANTrainSteps(1, 1)),
        config=get_run_config(check_point_dir, summary_steps, checkpoints_steps))

    train_spec = tf.estimator.TrainSpec(
        train_input_fn, max_steps=max_training_steps)
    final_exporter = tf.estimator.FinalExporter('final_exporter', serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=eval_steps, 
        exporters=[final_exporter])

    tf.estimator.train_and_evaluate(gan_estimator, train_spec, eval_spec)


parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_dir', type=str,
                    default='gs://gan-pipeline/checkpoints/')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_path', type=str,
                    default='gs://gan-pipeline/dataset/OCT2017/OCT2017/train/NORMAL/*')
parser.add_argument('--eval_image_path', type=str,
                    default='gs://gan-pipeline/dataset//OCT2017/OCT2017/test/NORMAL/*')
parser.add_argument('--summary_steps', type=int, default=100)
parser.add_argument('--checkpoints_steps', type=int, default=100)
parser.add_argument('--max_training_steps', type=int, default=10000)
parser.add_argument('--eval_steps', type=int, default=5000)

args = parser.parse_args()

check_point_dir = args.checkpoint_dir
batch_size = args.batch_size
image_path = args.image_path
eval_image_path = args.eval_image_path
summary_steps = args.summary_steps
checkpoints_steps = args.checkpoints_steps
max_training_steps = args.max_training_steps
eval_steps = args.eval_steps

end_to_end(check_point_dir, batch_size, image_path, summary_steps,
           checkpoints_steps, max_training_steps, eval_steps)
