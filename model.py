import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel, BertTokenizer, AdamWeightDecay

from utils.helpers import Vocab


class GEC:
    def __init__(self, max_len=128, confidence=0.0, min_error_prob=0.0,
                 learning_rate=1e-5,
                 vocab_path='data/output_vocab/',
                 verb_adj_forms_path='data/transform.txt',
                 bert_model='cl-tohoku/bert-base-japanese-v2',
                 pretrained_weights_path=None,
                 bert_trainable=True):
        self.max_len = max_len
        self.confidence = confidence
        self.min_error_prob = min_error_prob
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        vocab_labels_path = os.path.join(vocab_path, 'labels.txt')
        vocab_detect_path = os.path.join(vocab_path, 'detect.txt')
        self.vocab_labels = Vocab.from_file(vocab_labels_path)
        self.vocab_detect = Vocab.from_file(vocab_detect_path)
        self.model = self.get_model(bert_model, bert_trainable, learning_rate)
        if pretrained_weights_path:
            self.model.load_weights(pretrained_weights_path)
        self.transform = self.get_transforms(verb_adj_forms_path)

    def get_model(self, bert_model, bert_trainable=True, learning_rate=None):
        encoder = TFBertModel.from_pretrained(bert_model)
        encoder.bert.trainable = bert_trainable
        input_ids = layers.Input(shape=(self.max_len,), dtype=tf.int32,
            name='input_ids')
        attention_mask = input_ids != 0
        embedding = encoder(input_ids, attention_mask=attention_mask,
            training=bert_trainable)[0]
        n_labels = len(self.vocab_labels)
        n_detect = len(self.vocab_detect)
        labels_probs = layers.Dense(n_labels, activation='softmax',
            name='labels_probs')(embedding)
        detect_probs = layers.Dense(n_detect, activation='softmax',
            name='detect_probs')(embedding)
        model = keras.Model(
            inputs=input_ids,
            outputs=[labels_probs, detect_probs]
        )
        losses = [keras.losses.SparseCategoricalCrossentropy(),
                  keras.losses.SparseCategoricalCrossentropy()]
        optimizer = AdamWeightDecay(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=losses,
            weighted_metrics=['sparse_categorical_accuracy'])
        return model

    def predict(self, input_ids):
        labels_probs, detect_probs = self.model(input_ids, training=False)

        # get maximum INCORRECT probability across tokens for each sequence
        incorr_index = self.vocab_detect['INCORRECT']
        mask = tf.cast(input_ids != 0, tf.float32)
        error_probs = detect_probs[:, :, incorr_index] * mask
        max_error_probs = tf.math.reduce_max(error_probs, axis=-1)

        # boost $KEEP probability by self.confidence
        if self.confidence > 0:
            keep_index = self.vocab_labels['$KEEP']
            prob_change = np.zeros(labels_probs.shape[2])
            prob_change[keep_index] = self.confidence
            B = labels_probs.shape[0]
            S = labels_probs.shape[1]
            prob_change = tf.reshape(tf.tile(prob_change, [B * S]), [B, S, -1])
            labels_probs += prob_change

        output_dict = {
            'labels_probs': labels_probs.numpy(),  # (B, S, n_labels)
            'detect_probs': detect_probs.numpy(),  # (B, S, n_detect)
            'max_error_probs': max_error_probs.numpy(),  # (B,)
        }

        # get decoded text labels
        for namespace in ['labels', 'detect']:
            vocab = getattr(self, f'vocab_{namespace}')
            probs = output_dict[f'{namespace}_probs']
            decoded_batch = []
            for seq in probs:
                argmax_idx = np.argmax(seq, axis=-1)
                tags = [vocab[i] for i in argmax_idx]
                decoded_batch.append(tags)
            output_dict[namespace] = decoded_batch

        return output_dict

    def correct(self, sentences, max_iter=10):
        single = isinstance(sentences, str)
        cur_sentences = [sentences] if single else sentences
        for i in range(max_iter):
            new_sentences = self.correct_once(cur_sentences)
            if cur_sentences == new_sentences:
                break
            cur_sentences = new_sentences
        return cur_sentences[0] if single else cur_sentences

    def correct_once(self, sentences):
        input_dict = self.tokenizer(sentences, add_special_tokens=True,
            padding='max_length', max_length=self.max_len, return_tensors='tf')
        output_dict = self.predict(input_dict['input_ids'])
        labels = output_dict['labels']
        labels_probs = tf.math.reduce_max(
            output_dict['labels_probs'], axis=-1).numpy()
        new_sentences = []
        for i, sentence in enumerate(sentences):
            max_error_prob = output_dict['max_error_probs'][i]
            if max_error_prob < self.min_error_prob:
                new_sentences.append(sentence)
                continue
            input_ids = input_dict['input_ids'][i].numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            mask = input_dict['attention_mask'][i].numpy()
            for j in range(len(tokens)):
                if not mask[j]:
                    tokens[j] = ''
                elif labels_probs[i][j] < self.min_error_prob:
                    continue
                elif labels[i][j] in ['[PAD]', '[UNK]', '$KEEP']:
                    continue
                elif labels[i][j] == '$DELETE':
                    tokens[j] = ''
                elif labels[i][j].startswith('$APPEND_'):
                    tokens[j] += ' ' + labels[i][j].replace('$APPEND_', '')
                elif labels[i][j].startswith('$REPLACE_'):
                    tokens[j] = labels[i][j].replace('$REPLACE_', '')
                elif labels[i][j].startswith('$TRANSFORM_'):
                    transform_op = labels[i][j].replace('$TRANSFORM_', '')
                    key = f'{tokens[j]}_{transform_op}'
                    if key in self.transform:
                        tokens[j] = self.transform[key]
            tokens = ' '.join(tokens).split()
            tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
            new_sentence = self.tokenizer.convert_tokens_to_string(tokens)
            new_sentence = new_sentence.replace(' ', '')
            new_sentences.append(new_sentence)
        return new_sentences

    def get_transforms(self, verb_adj_forms_path):
        decode = {}
        with open(verb_adj_forms_path, 'r', encoding='utf-8') as f:
            for line in f:
                words, tags = line.split(':')
                tags = tags.strip()
                word1, word2 = words.split('_')
                tag1, tag2 = tags.split('_')
                decode_key = f'{word1}_{tag1}_{tag2}'
                if decode_key not in decode:
                    decode[decode_key] = word2
        return decode
