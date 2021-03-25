"""GAN tf.estimator model function"""

import tensorflow as tf
import os
# from trainer import metrics
import tensorflow.layers as layers
import tensorflow.contrib.gan as tfgan
from tensorflow.contrib.gan.python.eval import summaries as gansummaries
from tensorflow.contrib.gan.python.losses.python import losses_impl as ganloss
from tensorflow.contrib.gan.python.namedtuples import GANTrainSteps
import collections

def gen_res_block(tensor):
  conv_res = layers.conv2d(tensor,filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
  batch_norm_res = layers.batch_normalization(inputs=conv_res)
  relu_res = tf.nn.relu(batch_norm_res)
  conv_res_2 = layers.conv2d(relu_res, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
  batch_norm_res_2 = layers.batch_normalization(inputs=conv_res_2)
  res_1 = tensor + batch_norm_res_2  
  return res_1

def sub_pixel_block(batch_size,tensor):
  conv = layers.conv2d(
        inputs=tensor, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')
  dep_to_space = tf.nn.depth_to_space(conv, 2)
  batch_norm = layers.batch_normalization(dep_to_space)
  relu = tf.nn.relu(batch_norm)
  return relu

def dis_res_block(tensor, num_filters, kernelsize):
  conv_1 = layers.conv2d(tensor, filters=num_filters, kernel_size=kernelsize, strides=(1, 1), padding='same')
  lrelu_1 = tf.nn.leaky_relu(conv_1)
  ## resblock_sub_1
  conv_2 = layers.conv2d(inputs=lrelu_1, filters=num_filters, kernel_size=kernelsize, strides=(1, 1), padding='same' )
  lrelu_2 = tf.nn.leaky_relu(conv_2)
  conv_3 = layers.conv2d(inputs=lrelu_2, filters=num_filters, kernel_size=kernelsize, strides=(1, 1), padding='same' )
  layers_add_1 = lrelu_1 + conv_3
  lrelu_3 = tf.nn.leaky_relu(layers_add_1)
  ## resblock_sub_2
  conv_4 = layers.conv2d(inputs=lrelu_3, filters=num_filters, kernel_size=kernelsize, strides=(1, 1), padding='same' )
  lrelu_4 = tf.nn.leaky_relu(conv_4)
  conv_5 = layers.conv2d(inputs=lrelu_4, filters=num_filters, kernel_size=kernelsize, strides=(1, 1), padding='same' )
  layers_add_2 = lrelu_3 + conv_5
  res_out = tf.nn.leaky_relu(layers_add_2)
  return res_out

  #### GAN model ####
def get_generator_fn(batch_size):
  def generator_fn(input_features, mode):

    noise = tf.concat([input_features['noise'], input_features['features']], 1)
    noise = tf.reshape(noise, shape = (batch_size, 162))

    dense_1 = layers.dense(inputs=noise, units=64*16*16)
    reshape_1 = tf.reshape(dense_1, shape=(batch_size,16,16, 64))
    batch_norm_1 = layers.batch_normalization(inputs=reshape_1)
    relu_1 = tf.nn.relu(batch_norm_1)

    ##residual blocks
    res_layer1 = gen_res_block(relu_1)
    res_layer2 = gen_res_block(res_layer1)
    res_layer3 = gen_res_block(res_layer2)
    res_layer4 = gen_res_block(res_layer3)
    res_layer5 = gen_res_block(res_layer4)
    res_layer6 = gen_res_block(res_layer5)
    res_layer7 = gen_res_block(res_layer6)
    res_layer8 = gen_res_block(res_layer7)
    res_layer9 = gen_res_block(res_layer8)
    res_layer10 = gen_res_block(res_layer9)
    res_layer11 = gen_res_block(res_layer10)
    res_layer12 = gen_res_block(res_layer11)
    res_layer13 = gen_res_block(res_layer12)
    res_layer14 = gen_res_block(res_layer13)
    res_layer15 = gen_res_block(res_layer14)
    res_layer16 = gen_res_block(res_layer15)

  
    batch_norm_2 = layers.batch_normalization(inputs=res_layer16)
    relu_2 = tf.nn.relu(batch_norm_2)
    layers_add = relu_1+relu_2
    
    ##Subpixel block (make 3 times)
    sp_block1 = sub_pixel_block(batch_size, layers_add)
    sp_block2 = sub_pixel_block(batch_size, sp_block1)
    sp_block3 = sub_pixel_block(batch_size, sp_block2)
  

    conv_2 = layers.conv2d(
        inputs=sp_block3, filters=3, kernel_size=(9, 9), strides=(1, 1), padding='same')
    tanh_1 = tf.nn.tanh(conv_2)
    print(tanh_1)
    return tanh_1
  return generator_fn

def get_discriminator_fn(batch_size):
  def discriminator_fn(image, input_features):

    res_block_1 = dis_res_block(image, 32, 4)
    res_block_2 = dis_res_block(res_block_1, 64, 4)
    res_block_3 = dis_res_block(res_block_2, 128, 4)
    res_block_4 = dis_res_block(res_block_3, 256, 3)
    res_block_5 = dis_res_block(res_block_4, 512, 3)

    conv_1 = layers.conv2d(res_block_5, filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same')
    lrelu_1 = tf.nn.leaky_relu(conv_1, alpha=0.2)
    print(lrelu_1)

    reshape_1 = tf.reshape(lrelu_1, shape = (batch_size,64*64*32))
    print(reshape_1)
    fc1 = layers.dense(reshape_1, 1)
    # fc2 = layers.dense(reshape_1, 34)

    return fc1
  return discriminator_fn


# def get_generator_loss():
#   def combined_generator_loss(
#       gan_model,
#       perceptual_loss_order=2,
#       weight_factor=1.0,
#       gradient_ratio=None,
#       num_comparisons=10,
#       add_summaries=True):

  
#     discriminator_gen_outputs = gan_model.discriminator_gen_outputs
#     hr_image = gan_model.real_data
#     gen_image = gan_model.generated_data
#     lr_image = gan_model.generator_inputs['image']
#     tags = gan_model.generator_inputs['features']
#     # tags = tf.cast(tags, tf.int32)
#     print(discriminator_gen_outputs['label'])

#     adversarial_loss = ganloss.modified_generator_loss(
#         discriminator_gen_outputs=discriminator_gen_outputs['label'])
  
#     perceptual_loss = tf.norm(
#         discriminator_gen_outputs['tags'] - tags,
#         ord=perceptual_loss_order)

#     combined_adversarial_loss = ganloss.combine_adversarial_loss(
#         perceptual_loss,
#         adversarial_loss,
#         weight_factor=weight_factor,
#         gradient_ratio=gradient_ratio)

#     return combined_adversarial_loss
#   return combined_generator_loss


# def get_discriminator_loss():
#   def combined_discriminator_loss(
#       gan_model,
#       perceptual_loss_order=2,
#       weight_factor=1.0,
#       gradient_ratio=None,
#       num_comparisons=10,
#       add_summaries=True):
  
#     discriminator_gen_outputs = gan_model.discriminator_gen_outputs
#     discriminator_real_outputs = gan_model.discriminator_real_outputs
#     hr_image = gan_model.real_data
#     gen_image = gan_model.generated_data
#     lr_image = gan_model.generator_inputs['image']
#     tags = gan_model.generator_inputs['features']
#     # tags = tf.cast(tags, tf.int32)

#     tf.summary.image('Real_Image', hr_image)
#     tf.summary.image('Generated_Image', gen_image)



#     adversarial_loss = ganloss.modified_discriminator_loss(
#         discriminator_real_outputs=discriminator_real_outputs['label'],
#         discriminator_gen_outputs=discriminator_gen_outputs['label'],
#         add_summaries=add_summaries)

#     classifier_loss = tf.norm(
#         discriminator_gen_outputs['tags'] - tags,
#         ord=perceptual_loss_order)


#     # grad_penalty = 
#     combined_adversarial_loss = ganloss.combine_adversarial_loss(
#         classifier_loss,
#         adversarial_loss,
#         weight_factor=weight_factor,
#         gradient_ratio=gradient_ratio)

#     return combined_adversarial_loss
#   return combined_discriminator_loss



def get_estimator(params):
  run_config = tf.estimator.RunConfig(
        model_dir=params.job_dir,
        save_summary_steps=100,
        save_checkpoints_steps=200)

  gan_estimator = tfgan.estimator.GANEstimator(
        model_dir=params.job_dir,
        generator_fn=get_generator_fn(params.batch_size),
        discriminator_fn=get_discriminator_fn(params.batch_size),
        generator_loss_fn=tfgan.losses.modified_generator_loss,
        discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.00003, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.00001, 0.5),
        get_hooks_fn= tf.contrib.gan.get_sequential_train_hooks(train_steps=GANTrainSteps(1, 2)),
        config=run_config)

  return gan_estimator


