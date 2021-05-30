import os

import tensorflow as tf
from tensorflow.data import TFRecordDataset
from tensorflow.train import Features, Feature, Example, BytesList, Int64List
from tensorflow.io import (TFRecordWriter, TFRecordOptions, FixedLenFeature,
                           parse_single_example)


class Vocab:
    def __init__(self, words):
        self.id2word = words
        self.word2id = {word: i for i, word in enumerate(words)}

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
            return self.word2id[key]
        elif isinstance(key, int):
            return self.id2word[key]
        else:
            raise ValueError('Key must be str or int')


def write_dataset(path, examples):
    options = TFRecordOptions(compression_type='GZIP')
    with TFRecordWriter(path, options=options) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def read_dataset(paths):
    return TFRecordDataset(paths, compression_type='GZIP').map(parse_example)


def create_example(tokens, edits, tokenizer, labels_vocab, detect_vocab,
                   max_tokens_len=1024):
    if len(tokens) > max_tokens_len:
        print(f'Truncated {len(tokens)} tokens to {max_tokens_len} tokens')
        tokens = tokens[:max_tokens_len]
        edits = edits[:max_tokens_len]
    token_ids = [0] * max_tokens_len
    attention_mask = [0] * max_tokens_len
    label_ids = [0] * max_tokens_len
    detect_ids = [0] * max_tokens_len

    n = min(len(tokens), max_tokens_len)
    token_ids[:n] = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask[:n] = [1] * n
    label_ids[:n] = [labels_vocab[e[0]] for e in edits]
    corr_idx = detect_vocab['CORRECT']
    incorr_idx = detect_vocab['INCORRECT']
    detect_ids[:n] = [corr_idx if e == '$KEEP' else incorr_idx
                      for e in edits]

    assert len(token_ids) == max_tokens_len
    assert len(attention_mask) == max_tokens_len
    assert len(label_ids) == max_tokens_len
    assert len(detect_ids) == max_tokens_len

    feature = {
        'token_ids': int64_list_feature(token_ids),
        'att_mask': int64_list_feature(attention_mask),
        'label_ids': int64_list_feature(label_ids),
        'detect_ids': int64_list_feature(detect_ids)
    }
    return Example(features=Features(feature=feature))


def parse_example(example, max_tokens_len=1024):
    feature_desc = {
        'token_ids': FixedLenFeature([max_tokens_len], tf.int64),
        'att_mask': FixedLenFeature([max_tokens_len], tf.int64),
        'label_ids': FixedLenFeature([max_tokens_len], tf.int64),
        'detect_ids': FixedLenFeature([max_tokens_len], tf.int64)
    }
    e = parse_single_example(example, feature_desc)
    return (e['token_ids'], e['att_mask']), (e['label_ids'], e['detect_ids'])


def int64_list_feature(value):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return Feature(int64_list=Int64List(value=value))
