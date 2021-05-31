import argparse
import os

import tensorflow as tf

from model import GEC
from utils.helpers import read_dataset


def train(corpora_dir, output_dir, pretrained_dir, batch_size, n_epochs,
          dev_size, filename='edit_tagged_sentences.tfrec.gz'):
    files = [os.path.join(root, filename)
             for root, dirs, files in os.walk(corpora_dir)
             if filename in files]
    dataset = read_dataset(files).shuffle(buffer_size=1024)
    print(dataset)
    print('Loaded dataset')

    dev_i = int(dev_size * 100)
    train_set = dataset.enumerate().filter(lambda i, x: i % 100 >= dev_i).map(
        lambda i, x: x).batch(batch_size)
    dev_set = dataset.enumerate().filter(lambda i, x: i % 100 < dev_i).map(
        lambda i, x: x).batch(batch_size)
    print(f'Split dataset into train/dev = {100-dev_i}/{dev_i}')

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        gec = GEC(pretrained_model_path=pretrained_dir)
    gec.model.fit(train_set, epochs=n_epochs)
    gec.model.save(output_dir)


def main(args):
    train(args.corpora_dir, args.output_dir, args.pretrained_dir,
          args.batch_size, args.n_epochs, args.dev_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora_dir',
                        help='Path to dataset folder',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to saved model output dir',
                        required=True)
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
    args = parser.parse_args()
    main(args)
