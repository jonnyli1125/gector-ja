import argparse
import os
import math

import tensorflow as tf

from model import GEC
from utils.helpers import read_dataset


def train(corpora_dir, output_dir, vocab_dir, transforms_file, pretrained_dir,
          batch_size, n_epochs, dev_size, dataset_len,
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
             for root, dirs, files in os.walk(corpora_dir)
             if filename in files]
    dataset = read_dataset(files).shuffle(buffer_size=1024)
    if dataset_len:
        dataset_card = tf.data.experimental.assert_cardinality(dataset_len)
        dataset = dataset.apply(dataset_card)
    print(dataset, dataset.cardinality().numpy())
    print('Loaded dataset')

    dev_i = int(dev_size * 100)
    train_set = dataset.enumerate().filter(lambda i, x: i % 100 >= dev_i).map(
        lambda i, x: x)
    dev_set = dataset.enumerate().filter(lambda i, x: i % 100 < dev_i).map(
        lambda i, x: x)
    if dataset_len:
        dev_len = math.ceil(dataset_len / 100) * dev_i
        train_len = dataset_len - dev_len
        dev_card = tf.data.experimental.assert_cardinality(dev_len)
        train_card = tf.data.experimental.assert_cardinality(train_len)
        train_set = train_set.apply(train_card)
        dev_set = dev_set.apply(dev_card)
        print(train_set.cardinality().numpy(), dev_set.cardinality().numpy())
    train_set = train_set.batch(batch_size)
    dev_set = dev_set.batch(batch_size)
    print(f'Split dataset into train/dev = {100-dev_i}/{dev_i}')

    if tpu:
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        gec = GEC(vocab_path=vocab_dir, pretrained_model_path=pretrained_dir,
                  verb_adj_forms_path=transforms_file)
    gec.model.fit(train_set, epochs=n_epochs, validation_data=dev_set)
    gec.model.save(output_dir)


def main(args):
    train(args.corpora_dir, args.output_dir, args.vocab_dir,
          args.transforms_file, args.pretrained_dir, args.batch_size,
          args.n_epochs, args.dev_size, args.dataset_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora_dir',
                        help='Path to dataset folder',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to saved model output folder',
                        required=True)
    parser.add_argument('-v', '--vocab_dir',
                        help='Path to output vocab folder',
                        default='./data/output_vocab')
    parser.add_argument('-t', '--transforms_file',
                        help='Path to verb/adj transforms file',
                        default='./data/transform.txt')
    parser.add_argument('-p', '--pretrained_dir',
                        help='Path to pretrained model dir')
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of samples per batch',
                        default=32)
    parser.add_argument('-e', '--n_epochs', type=int,
                        help='Number of epochs',
                        default=10)
    parser.add_argument('-d', '--dev_size', type=float,
                        help='Percent of whole dataset to use for dev set',
                        default=0.01)
    parser.add_argument('-l', '--dataset_len', type=int,
                        help='Cardinality of dataset')
    args = parser.parse_args()
    main(args)
