import argparse
import os
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np
from transformers import AdamWeightDecay
from sklearn.metrics import classification_report

from model import GEC
from utils.helpers import read_dataset


AUTO = tf.data.AUTOTUNE


def train(corpora_dir, output_weights_path, vocab_dir, transforms_file,
          pretrained_weights_path, batch_size, n_epochs, dev_ratio, dataset_len,
          dataset_ratio, bert_trainable, learning_rate, class_weight_path,
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
        if class_weight_path:
            with open(class_weight_path) as f:
                class_weight = json.load(f)
            losses = [WeightedSCCE(w) for w in class_weight]
            print('Using weighted SCCE loss')
        else:
            losses = [keras.losses.SparseCategoricalCrossentropy(),
                keras.losses.SparseCategoricalCrossentropy()]
        optimizer = AdamWeightDecay(learning_rate=learning_rate)
        metrics = [keras.metrics.SparseCategoricalAccuracy()]
        gec.model.compile(optimizer=optimizer, loss=losses,
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
    print('Confusion matrices:')
    y_pred = gec.model.predict(dev_set)
    for i, y_pred in enumerate(gec.model.predict(dev_set)):
        y_pred = tf.reshape(tf.math.argmax(y_pred, axis=-1), [-1])
        y_true = np.concatenate([y[i] for x, y, w in dev_set], axis=0).flatten()
        weights = y_true != 0
        print(tf.math.confusion_matrix(y_true, y_pred, weights=weights).numpy())
        print(classification_report(y_true, y_pred, sample_weight=weights))

class WeightedSCCE(keras.losses.Loss):
    def __init__(self, class_weight, from_logits=False, name='weighted_scce'):
        if class_weight is None or all(v == 1. for v in class_weight):
            self.class_weight = None
        else:
            self.class_weight = tf.convert_to_tensor(class_weight,
                dtype=tf.float32)
        self.reduction = keras.losses.Reduction.NONE
        self.unreduced_scce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, name=name,
            reduction=self.reduction)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.unreduced_scce(y_true, y_pred, sample_weight)
        if self.class_weight is not None:
            weight_mask = tf.gather(self.class_weight, y_true)
            loss = tf.math.multiply(loss, weight_mask)
        return loss


def main(args):
    train(args.corpora_dir, args.output_weights_path, args.vocab_dir,
          args.transforms_file, args.pretrained_weights_path, args.batch_size,
          args.n_epochs, args.dev_ratio, args.dataset_len, args.dataset_ratio,
          args.bert_trainable, args.learning_rate, args.class_weight_path)


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
    parser.add_argument('-cw', '--class_weight_path',
                        help='Path to class weight file')
    args = parser.parse_args()
    main(args)
