"""Autoencoder tf.estimator model function"""

import tensorflow as tf
import os
# from trainer import metrics
import tensorflow.layers as layers
import tensorflow.contrib.gan as tfgan
from tensorflow.contrib.gan.python.namedtuples import GANTrainSteps


def encoder_model_fn(features, labels, mode, params):
  # global graph
  # graph = tf.reset_default_graph()
  # with graph.as_default():

  # feature columns for input_layer
  feature_columns = [tf.feature_column.numeric_column('image')]

  input_layer = tf.feature_column.input_layer(features=features,
                                                feature_columns=feature_columns)


  is_training = (mode == tf.estimator.ModeKeys.TRAIN)


  # drop_out_1 = tf.nn.dropout(input_layer)
  reshape_1 = tf.reshape(
            input_layer, shape=(-1, 64, 64, 3))
  conv_1 = layers.conv2d(
            inputs=reshape_1, filters=3, kernel_size=(3, 3), padding='same')
  res_1 = res_block(conv_1)
  res_2 = res_block(res_1)
  res_3 = res_block(res_2)
  res_4 = res_block(res_3)
  dense_1 = layers.dense(inputs=res_4, units=10)
  encoding_output = tf.nn.tanh(dense_1)
  image_output = get_images(encoding_output)
  if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'encoding': encoding_output,
          'image': image_output
      }
      export_outputs = {
          'predict': tf.estimator.export.PredictOutput(predictions)
      }
      return tf.estimator.EstimatorSpec(mode,
                                        predictions=predictions,
                                        export_outputs=export_outputs)

  # loss: sum of mse and regularization loss
  else:
      loss = tf.losses.mean_squared_error(tf.squeeze(input_layer), image_output)
      loss = loss + tf.losses.get_regularization_loss()

      optimizer = tf.train.AdamOptimizer(0.0001)
      train_op = optimizer.minimize(loss=loss,
                                    global_step=tf.train.get_global_step())
      eval_metric_ops = {
          "rmse": tf.metrics.root_mean_squared_error(
              tf.squeeze(input_layer), image_output)
      }
      estimator_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                  loss=loss,
                                                  train_op=train_op,
                                                  eval_metric_ops=eval_metric_ops)
      return estimator_spec

# Estimator wrapper
def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=encoder_model_fn,
                                       params=hparams,
                                       config=run_config)

    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")
    return estimator


def res_block(inputlayer):
  pool_1 =layers.average_pooling2d(
            inputlayer, pool_size=(2, 2), strides=(1, 1))
  conv_2 = layers.conv2d(
                inputs=pool_1, filters=3, kernel_size=(3, 3), padding='same')
  
  paddings = tf.constant([[0,0],[0, 1,], [0, 1],[0,0]])
  padded_conv_2 = tf.pad(conv_2, paddings, "CONSTANT")
  print("")
  print('padded_conv_2', padded_conv_2)
  batch_norm_1 = layers.batch_normalization(inputs=inputlayer)
  relu_1 = tf.nn.relu(batch_norm_1)
  conv_3 = layers.conv2d(
                inputs=relu_1, filters=3, kernel_size=(3, 3), padding='same')
  batch_norm_2 = layers.batch_normalization(inputs=conv_3)
  relu_2 = tf.nn.relu(batch_norm_2)
  conv_4 = layers.conv2d(
                inputs=relu_2, filters=3, kernel_size=(3, 3), padding='same')
  print('conv_4',conv_4)
  print("")
  res_out_1 = padded_conv_2+conv_4
  return res_out_1




  #### GAN model ####
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


def preprocess_image(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [64, 64])
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


def _get_predict_input_fn(encoder_output):
    def predict_input_fn():
        return encoder_output
    return predict_input_fn


def get_run_config(check_point_dir, summary_steps, checkpoints_steps):
    run_config = tf.estimator.RunConfig(
        model_dir=check_point_dir,
        save_summary_steps=summary_steps,
        save_checkpoints_steps=checkpoints_steps)
    return run_config

def get_images(encoder_output):
  # tf.get_default_session().close()
  # global graph
  # graph = tf.reset_default_graph()
  # with graph.as_default():

  train_input_fn = _get_train_input_fn(gan_image_path, gan_batch_size)
  eval_input_fn = _get_train_input_fn(gan_eval_image_path, gan_batch_size)
  gan_estimator = tfgan.estimator.GANEstimator(
    model_dir=gan_check_point_dir,
    generator_fn=get_generator_fn(gan_batch_size),
    discriminator_fn=discriminator_fn,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    generator_optimizer=tf.train.AdamOptimizer(0.00003, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.00001, 0.5),
    get_hooks_fn= tf.contrib.gan.get_sequential_train_hooks(train_steps=GANTrainSteps(1, 1)),
    config=get_run_config(gan_check_point_dir, gan_summary_steps, gan_checkpoints_steps))

  pred_in_fn = _get_predict_input_fn(encoder_output=encoder_output)
  image_output = gan_estimator.predict(pred_in_fn)
  for img in image_output:
    print("*****output")
    print(img)
  return image_output


gan_check_point_dir = 'gs://gan-pipeline/checkpoints/fAnogan11'
gan_batch_size = 64
gan_image_path = 'gs://gan-pipeline/dataset/OCT2017/OCT2017/train/NORMAL/*'
gan_eval_image_path = 'gs://gan-pipeline/dataset//OCT2017/OCT2017/test/NORMAL/*'
gan_summary_steps = 100
gan_checkpoints_steps = 100
