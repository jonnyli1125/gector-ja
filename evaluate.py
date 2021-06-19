import os
import argparse

import tensorflow as tf
from fugashi import Tagger
from sklearn.metrics import classification_report, fbeta_score

from model import GEC
from utils.edits import EditTagger
from utils.helpers import write_dataset, read_dataset


def main(weights_path, vocab_dir, transforms_file, source_path, target_path,
         tfrec_path):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print('TPUs: ', tf.config.list_logical_devices('TPU'))
    except (ValueError, KeyError) as e:
        tpu = None
    if not tf.io.gfile.exists(tfrec_path):
        dirname = os.path.dirname(tfrec_path)
        if not tf.io.gfile.exists(dirname):
            tf.io.gfile.makedirs(dirname)
        with tf.io.gfile.GFile(source_path, 'r') as f:
            source_lines = f.readlines()
        with tf.io.gfile.GFile(target_path, 'r') as f:
            target_lines = f.readlines()
        edit_tagger = EditTagger(verb_adj_forms_path=transforms_file,
            vocab_detect_path=os.path.join(vocab_dir, 'detect.txt'),
            vocab_labels_path=os.path.join(vocab_dir, 'labels.txt'))
        rows = []
        for source, target in zip(source_lines, target_lines):
            if not source or not target:
                continue
            edit_levels = edit_tagger(source, target, levels=True)
            rows.extend(edit_levels)
        write_dataset(tfrec_path, rows)
        print(f'Wrote {len(rows)} examples to {tfrec_path}')

    dataset = read_dataset(tfrec_path)
    X = []
    Y = [[], []]
    W = []
    for x, (y_l, y_d), w in dataset:
        X.append(x)
        Y[0].append(y_l)
        Y[1].append(y_d)
        W.append(w)
    X = tf.stack(X)
    Y[0] = tf.stack(Y[0])
    Y[1] = tf.stack(Y[1])
    W = tf.stack(W).numpy().flatten()

    if tpu:
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        gec = GEC(vocab_path=vocab_dir, verb_adj_forms_path=transforms_file,
                  pretrained_weights_path=weights_path)
        gec.model.compile()

    pred = gec.model.predict(X)
    for i, y_pred in enumerate(pred):
        y_pred = tf.math.argmax(y_pred, axis=-1).numpy().flatten()
        y_true = Y[i].numpy().flatten()
        print(classification_report(y_true, y_pred, sample_weight=W))
        print('F0.5', fbeta_score(y_true, y_pred, beta=0.5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights',
                        help='Path to model weights',
                        required=True)
    parser.add_argument('-v', '--vocab_dir',
                        help='Path to output vocab folder',
                        default='./data/output_vocab')
    parser.add_argument('-f', '--transforms_file',
                        help='Path to verb/adj transforms file',
                        default='./data/transform.txt')
    parser.add_argument('-s', '--source_path',
                        help='Path to source lines of evaluation corpus',
                        required=True)
    parser.add_argument('-t', '--target_path',
                        help='Path to target lines of evaluation corpus',
                        required=True)
    parser.add_argument('-e', '--tfrec_path',
                        help='Path to edit tagged lines of evaluation corpus',
                        required=True)
    args = parser.parse_args()
    main(args.weights, args.vocab_dir, args.transforms_file, args.source_path,
         args.target_path, args.tfrec_path)
