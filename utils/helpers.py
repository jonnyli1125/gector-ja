import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import TFRecordDataset, AUTOTUNE
from tensorflow.train import Features, Feature, Example, BytesList, Int64List
from tensorflow.io import (TFRecordWriter, TFRecordOptions, FixedLenFeature,
                           parse_single_example)


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


class Vocab:
    def __init__(self, words):
        self.id2word = words
        self.word2id = {word: i for i, word in enumerate(words)}
        self.unk_id = self.word2id['[UNK]'] if '[UNK]' in self.word2id else -1

    @classmethod
    def from_file(cls, file):
        if not os.path.exists(file):
            raise ValueError(f'Vocab file {file} does not exist')
        words = []
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    words.append(line)
        return cls(words)

    def __len__(self):
        return len(self.id2word)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.word2id.get(key, self.unk_id)
        else:
            return self.id2word[key]


def write_dataset(path, examples):
    options = TFRecordOptions(compression_type='GZIP')
    with TFRecordWriter(path, options=options) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def read_dataset(paths):
    return TFRecordDataset(paths, compression_type='GZIP',
        num_parallel_reads=AUTOTUNE).map(parse_example,
        num_parallel_calls=AUTOTUNE)


def create_example(tokens, edits, tokenizer, labels_vocab, detect_vocab,
                   max_tokens_len=128):
    if len(tokens) > max_tokens_len:
        tokens = tokens[:max_tokens_len]
        edits = edits[:max_tokens_len]
    token_ids = [0] * max_tokens_len
    label_ids = [0] * max_tokens_len
    detect_ids = [0] * max_tokens_len

    n = min(len(tokens), max_tokens_len)
    token_ids[:n] = tokenizer.convert_tokens_to_ids(tokens)
    label_ids[:n] = [labels_vocab[e] for e in edits]
    corr_idx = detect_vocab['CORRECT']
    incorr_idx = detect_vocab['INCORRECT']
    detect_ids[:n] = [corr_idx if e == '$KEEP' else incorr_idx for e in edits]

    assert len(token_ids) == max_tokens_len
    assert len(label_ids) == max_tokens_len
    assert len(detect_ids) == max_tokens_len

    feature = {
        'token_ids': int64_list_feature(token_ids),
        'label_ids': int64_list_feature(label_ids),
        'detect_ids': int64_list_feature(detect_ids)
    }
    return Example(features=Features(feature=feature))


def parse_example(example, max_tokens_len=128):
    feature_desc = {
        'token_ids': FixedLenFeature([max_tokens_len], tf.int64),
        'label_ids': FixedLenFeature([max_tokens_len], tf.int64),
        'detect_ids': FixedLenFeature([max_tokens_len], tf.int64)
    }
    example = parse_single_example(example, feature_desc)
    token_ids = tf.cast(example['token_ids'], tf.int32)
    att_mask = token_ids != 0
    label_ids = example['label_ids']
    detect_ids = example['detect_ids']
    return token_ids, (label_ids, detect_ids), att_mask


def int64_list_feature(value):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return Feature(int64_list=Int64List(value=value))
