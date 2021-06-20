import os
import argparse

import tensorflow as tf
from fugashi import Tagger
from nltk.translate.gleu_score import corpus_gleu

from model import GEC


tagger = Tagger('-Owakati')


def tokenize(sentence):
    return [t.surface for t in tagger(sentence)]


def main(weights_path, vocab_dir, transforms_file, corpus_dir):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        print('TPUs: ', tf.config.list_logical_devices('TPU'))
    except (ValueError, KeyError) as e:
        tpu = None
    source_path = tf.io.gfile.glob(os.path.join(corpus_dir, '*.src'))[0]
    with tf.io.gfile.GFile(source_path, 'r') as f:
        source_sents = [line for line in f.readlines() if line]
    reference_tokens = []
    for reference_path in tf.io.gfile.glob(os.path.join(corpus_dir, '*.ref*')):
        with tf.io.gfile.GFile(reference_path, 'r') as f:
            tokens = [tokenize(line) for line in f.readlines() if line]
            reference_tokens.append(tokens)
    reference_tokens = list(zip(*reference_tokens))
    print(f'Loaded {len(source_sents)} src, {len(reference_tokens)} ref')

    if tpu:
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        gec = GEC(vocab_path=vocab_dir, verb_adj_forms_path=transforms_file,
                  pretrained_weights_path=weights_path)

    pred_tokens = []
    source_batches = [source_sents[i:i + 64]
                      for i in range(0, len(source_sents), 64)]
    for i, source_batch in enumerate(source_batches):
        print(f'Predict batch {i+1}/{len(source_batches)}')
        pred_batch = gec.correct(source_batch)
        pred_batch_tokens = [tokenize(sent) for sent in pred_batch]
        pred_tokens.extend(pred_batch_tokens)
    print('Corpus GLEU', corpus_gleu(reference_tokens, pred_tokens))


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
    parser.add_argument('-c', '--corpus_dir',
                        help='Path to directory of TMU evaluation corpus',
                        required=True)
    args = parser.parse_args()
    main(args.weights, args.vocab_dir, args.transforms_file, args.corpus_dir)
