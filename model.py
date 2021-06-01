import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFAutoModel, AutoTokenizer

from utils.helpers import Vocab
from utils.edits import EditTagger


class GEC:
    def __init__(self, max_len=512, confidence=0.0, min_error_prob=0.0,
                 vocab_path='data/output_vocab/',
                 verb_adj_forms_path='data/transform.txt',
                 bert_model='cl-tohoku/bert-base-japanese-v2',
                 pretrained_model_path=None):
        self.max_len = max_len
        self.confidence = confidence
        self.min_error_prob = min_error_prob
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        vocab_labels_path = os.path.join(vocab_path, 'labels.txt')
        vocab_detect_path = os.path.join(vocab_path, 'detect.txt')
        self.vocab_labels = Vocab.from_file(vocab_labels_path)
        self.vocab_detect = Vocab.from_file(vocab_detect_path)
        if pretrained_model_path:
            self.model = keras.models.load_model(pretrained_model_path)
        else:
            self.model = self.create_model(bert_model)
        self.edit_tagger = EditTagger(tokenizer=self.tokenizer,
            vocab_labels_path=vocab_labels_path,
            vocab_detect_path=vocab_detect_path,
            verb_adj_forms_path=verb_adj_forms_path)

    def create_model(self, bert_model):
        encoder = TFAutoModel.from_pretrained(bert_model)
        encoder.bert.trainable = False
        input_ids = layers.Input(shape=(self.max_len,), dtype=tf.int32,
            name='input_ids')
        attention_mask = layers.Input(shape=(self.max_len,), dtype=tf.int32,
            name='attention_mask')
        embedding = encoder(input_ids, attention_mask=attention_mask)[0]
        labels_probs = layers.Dense(len(self.vocab_labels),
            activation='softmax', name='labels_probs')(embedding)
        detect_probs = layers.Dense(len(self.vocab_detect),
            activation='softmax', name='detect_probs')(embedding)
        model = keras.Model(
            inputs=[input_ids, attention_mask],
            outputs=[labels_probs, detect_probs]
        )
        loss = keras.losses.SparseCategoricalCrossentropy()
        optimizer = keras.optimizers.Adam()
        metrics = [keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.Precision(), keras.metrics.Recall()]
        model.compile(optimizer=optimizer, loss=[loss, loss], metrics=metrics)
        return model

    def predict(self, input_dict):
        labels_probs, detect_probs = self.model(input_dict)

        # get maximum INCORRECT probability across tokens for each sequence
        incorr_index = self.vocab_detect['INCORRECT']
        mask = input_dict['attention_mask']
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
            'labels_probs': labels_probs,  # (B, S, n_labels)
            'detect_probs': detect_probs,  # (B, S, n_detect)
            'max_error_probs': max_error_probs,  # (B,)
        }

        # get decoded text labels
        for namespace, probs in ['labels', 'detect']:
            vocab = getattr(self, f'vocab_{namespace}')
            probs = output_dict[f'{namespace}_probs'].numpy()
            decoded_batch = []
            for seq in probs:
                argmax_idx = np.argmax(seq, axis=-1)
                tags = [vocab[i] for i in argmax_idx]
                decoded_batch.append(tags)
            output_dict[namespace] = decoded_batch

        return output_dict

    def correct(self, sentence, max_iter=10):
        cur_sentence = sentence
        for i in range(max_iter):
            new_sentences = self.correct_once(cur_sentence)
            if cur_sentence == new_sentence:
                break
            cur_sentence = new_sentence
        return cur_sentence

    def correct_once(self, sentence):
        input_dict = self.tokenizer(sentence, add_special_tokens=True,
            padding='max_length', max_length=self.max_len,
            return_token_type_ids=False)
        output_dict = self.predict(input_dict)
        labels = output_dict['labels']
        labels_probs = tf.math.reduce_max(
            output_dict['labels_probs'], axis=-1)[0]
        max_error_prob = output_dict['max_error_probs'][0]
        if max_error_prob < self.min_error_prob:
            return sentence
        tokens = self.tokenizer.convert_ids_to_tokens(input_dict['input_ids'])
        mask = input_dict['attention_mask']
        for i in range(len(tokens)):
            if labels_probs[i] < self.min_error_prob:
                labels[i] = '$KEEP'
        new_tokens = self.edit_tagger.apply_edits(tokens, labels)
        new_sentence = self.edit_tagger.join_tokens(new_tokens)
        return new_sentence
