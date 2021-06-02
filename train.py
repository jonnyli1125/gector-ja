import argparse
import os
import math

import tensorflow as tf
from tensorflow import keras

from model import GEC
from utils.helpers import read_dataset


AUTO = tf.data.AUTOTUNE


def train(corpora_dir, output_weights_path, vocab_dir, transforms_file,
          pretrained_weights_path, batch_size, n_epochs, dev_ratio, dataset_len,
          dataset_ratio, bert_trainable, learning_rate,
          filename='edit_tagged_sentences.tfrec.gz'):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print('TPUs: ', tf.config.list_logical_devices('TPU'))
    except (ValueError, KeyError) as e:
        tpu = None
    files = [os.path.join(root, filename)
             for root, dirs, files in tf.io.gfile.walk(corpora_dir)
             if filename in files]
    dataset = read_dataset(files).shuffle(buffer_size=1024)
    if dataset_len:
        dataset_card = tf.data.experimental.assert_cardinality(dataset_len)
        dataset = dataset.apply(dataset_card)
    if 0 < dataset_ratio < 1:
        dataset_len = int(dataset_len * dataset_ratio)
        dataset = dataset.take(dataset_len)
    print(dataset, dataset.cardinality().numpy())
    print('Loaded dataset')

    dev_len = int(dataset_len * dev_ratio)
    train_set = dataset.skip(dev_len).prefetch(AUTO)
    dev_set = dataset.take(dev_len).prefetch(AUTO)
    print(train_set.cardinality().numpy(), dev_set.cardinality().numpy())
    print(f'Using {dev_ratio} of dataset for dev set')
    train_set = train_set.batch(batch_size, num_parallel_calls=AUTO)
    dev_set = dev_set.batch(batch_size, num_parallel_calls=AUTO)

    if tpu:
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        gec = GEC(vocab_path=vocab_dir, verb_adj_forms_path=transforms_file,
            pretrained_weights_path=pretrained_weights_path,
            bert_trainable=bert_trainable)
        loss = keras.losses.SparseCategoricalCrossentropy()
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        metrics = [keras.metrics.SparseCategoricalAccuracy()]
        gec.model.compile(optimizer=optimizer, loss=[loss, loss],
            metrics=metrics)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=output_weights_path + '_checkpoint',
        save_weights_only=True,
        monitor='val_labels_probs_sparse_categorical_accuracy',
        mode='max',
        save_best_only=True)
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='loss', patience=3)
    gec.model.fit(train_set, epochs=n_epochs, validation_data=dev_set,
        callbacks=[model_checkpoint_callback, early_stopping_callback])
    gec.model.save_weights(output_weights_path)


def main(args):
    train(args.corpora_dir, args.output_weights_path, args.vocab_dir,
          args.transforms_file, args.pretrained_weights_path, args.batch_size,
          args.n_epochs, args.dev_ratio, args.dataset_len, args.dataset_ratio,
          args.bert_trainable, args.learning_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora_dir',
                        help='Path to dataset folder',
                        required=True)
    parser.add_argument('-o', '--output_weights_path',
                        help='Path to save model weights to',
                        required=True)
    parser.add_argument('-v', '--vocab_dir',
                        help='Path to output vocab folder',
                        default='./data/output_vocab')
    parser.add_argument('-t', '--transforms_file',
                        help='Path to verb/adj transforms file',
                        default='./data/transform.txt')
    parser.add_argument('-p', '--pretrained_weights_path',
                        help='Path to pretrained model weights')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of samples per batch',
                        default=32)
    parser.add_argument('-e', '--n_epochs', type=int,
                        help='Number of epochs',
                        default=10)
    parser.add_argument('-d', '--dev_ratio', type=float,
                        help='Percent of whole dataset to use for dev set',
                        default=0.01)
    parser.add_argument('-l', '--dataset_len', type=int,
                        help='Cardinality of dataset')
    parser.add_argument('-r', '--dataset_ratio', type=float,
                        help='Percent of whole dataset to use',
                        default=1.0)
    parser.add_argument('-bt', '--bert_trainable',
                        help='Enable training for BERT encoder layers',
                        action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='Learning rate',
                        default=1e-5)
    args = parser.parse_args()
    main(args)
