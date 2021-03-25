## CMLE trainer for Dimensionality reduction

This tf.estimator trainer is optimized for CMLE training

Data: Input should be tfrecords holding key, 1D value pairs (encounter_id, word_embeddings)

Model: Autoencoder decoder


### Running training and eval

To run local training jobs
```angular2html
gcloud ai-platform local train\
          --module-name trainer.task \
          --package-path /home/neeti/Documents/GANs/Anime_Creation/Anime_gan_trainer/trainer \
          -- \
          --job_dir 'gs://gan-pipeline/checkpoints/local3' \
          --train_file gs://gan-pipeline/dataset/getchu_images/avatar_with_tag.list \
          --eval_file gs://gan-pipeline/dataset/getchu_images/avatar_with_tag.list \
          --batch_size 64 \
          --train_steps 5000

```
