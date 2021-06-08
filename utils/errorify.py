import re
import json
import traceback

import numpy as np
from numpy.random import choice, randint, uniform, binomial
from fugashi import Tagger


class Errorify:
    """Generate artificial errors in sentences."""

    def __init__(self, particle2freq_path='./data/particle_freq.json'):
        self.tagger = Tagger('-Owakati')
        self.error_prob = 1/8
        self.particles = ['は', 'と', 'を', 'の', 'で', 'が', 'に', 'から', 'へ',
                          'より', 'まで', 'って', 'て', 'や', 'か']

    def delete_error(self, token, feature):
        """Delete a random token or character."""
        i = randint(len(token)+1)  # pick a char or whole token
        if i == len(token):
            return ''
        else:
            return token[:i] + token[i+1:]

    def inflection_error(self, token, feature):
        """Misinflect a random verb/adj stem."""
        baseform = feature.orthBase or feature.lemma
        if not baseform:
            return token
        morphs = list(self.get_forms(baseform).values())
        if not morphs:
            return token
        return choice(morphs)

    def insert_error(self, token, feature):
        """Insert a random kanji, particle, or typo character."""
        new_word = choice(self.particles)
        return token + new_word

    def replace_error(self, token, feature):
        """Replace a random particle or kanji/verb/adj with same reading."""
        return choice(self.particles)

    def __call__(self, sentence):
        """Get sentence with artificially generated errors."""
        # need to this because fugashi has some weird bug
        tokens = [(t.surface, t.feature) for t in self.tagger(sentence)]
        error_tokens = []
        error_prob = self.error_prob
        for token, feature in tokens:
            if uniform() >= error_prob:
                error_tokens.append(token)
                error_prob += self.error_prob
                continue
            error_prob = self.error_prob
            if feature.pos2 in ['格助詞', '副助詞', '係助詞']:
                error_func = choice([self.insert_error, self.delete_error,
                                     self.replace_error], p=[0.1, 0.45, 0.45])
            elif feature.pos1 in ['動詞', '形容詞']:
                error_func = choice([self.insert_error, self.inflection_error],
                                    p=[0.1, 0.9])
            elif feature.pos2 not in ['数詞']:
                error_func = choice([self.insert_error, self.delete_error])
            else:
                error_func = lambda t, f: t
            error_token = error_func(token, feature)
            error_tokens.append(error_token)
        return ''.join(error_tokens)

    def get_forms(self, baseform):
        f = self.tagger(baseform)[0].feature
        # irregular verbs
        if f.orthBase == 'する' and f.lemma == '為る':
            return {
                'VB': 'する',  # plain (終止形)
                'VBI': 'し',  # imperfect (未然形)
                'VBC': 'し',  # conjunctive (連用形)
                'VBCG': 'し',  # conjunctive geminate (連用形-促音便)
                'VBP': 'しろ',  # imperative (命令系)
                'VBV': 'しよう',  # volitional (意志推量形)
                'VBS': 'する'  # stem/subword token
            }
        elif f.kanaBase == 'イク' or (f.orthBase and f.orthBase[-2:] == '行く'):
            forms = {
                'VB': 'く',  # plain (終止形)
                'VBI': 'か',  # imperfect (未然形)
                'VBC': 'き',  # conjunctive (連用形)
                'VBCG': 'っ',  # conjunctive geminate (連用形-促音便)
                'VBP': 'け',  # imperative (命令系)
                'VBV': 'こう',  # volitional (意志推量形)
                'VBS': 'く'  # stem/subword token
            }
        elif f.pos1 == '形容詞':  # i-adj
            forms = {
                'ADJ': 'い',  # plain (終止形)
                'ADJC': 'く',  # conjunctive (連用形)
                'ADJCG': 'かっ',  # conjunctive geminate (連用形-促音便)
                'ADJS': ''  # stem/subword token
            }
        elif '一段' in f.cType:  # ru-verbs
            forms = {
                'VB': 'る',  # plain (終止形)
                'VBI': '',  # imperfect (未然形)
                'VBC': '',  # conjunctive (連用形)
                'VBCG': '',  # conjunctive geminate (連用形-促音便)
                'VBP': 'ろ',  # imperative (命令系)
                'VBV': 'よう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'る':  # u-verbs from here
            forms = {
                'VB': 'る',  # plain (終止形)
                'VBI': 'ら',  # imperfect (未然形)
                'VBC': 'り',  # conjunctive (連用形)
                'VBCG': 'っ',  # conjunctive geminate (連用形-促音便)
                'VBP': 'れ',  # imperative (命令系)
                'VBV': 'ろう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'つ':
            forms = {
                'VB': 'つ',  # plain (終止形)
                'VBI': 'た',  # imperfect (未然形)
                'VBC': 'ち',  # conjunctive (連用形)
                'VBCG': 'っ',  # conjunctive geminate (連用形-促音便)
                'VBP': 'て',  # imperative (命令系)
                'VBV': 'とう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'う':
            forms = {
                'VB': 'う',  # plain (終止形)
                'VBI': 'わ',  # imperfect (未然形)
                'VBC': 'い',  # conjunctive (連用形)
                'VBCG': 'っ',  # conjunctive geminate (連用形-促音便)
                'VBP': 'え',  # imperative (命令系)
                'VBV': 'おう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'く':
            forms = {
                'VB': 'く',  # plain (終止形)
                'VBI': 'か',  # imperfect (未然形)
                'VBC': 'き',  # conjunctive (連用形)
                'VBCG': 'い',  # conjunctive geminate (連用形-促音便)
                'VBP': 'け',  # imperative (命令系)
                'VBV': 'こう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'ぐ':
            forms = {
                'VB': 'ぐ',  # plain (終止形)
                'VBI': 'が',  # imperfect (未然形)
                'VBC': 'ぎ',  # conjunctive (連用形)
                'VBCG': 'い',  # conjunctive geminate (連用形-促音便)
                'VBP': 'げ',  # imperative (命令系)
                'VBV': 'ごう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'す':
            forms = {
                'VB': 'す',  # plain (終止形)
                'VBI': 'さ',  # imperfect (未然形)
                'VBC': 'し',  # conjunctive (連用形)
                'VBCG': 'し',  # conjunctive geminate (連用形-促音便)
                'VBP': 'せ',  # imperative (命令系)
                'VBV': 'そう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'む':
            forms = {
                'VB': 'む',  # plain (終止形)
                'VBI': 'ま',  # imperfect (未然形)
                'VBC': 'み',  # conjunctive (連用形)
                'VBCG': 'ん',  # conjunctive geminate (連用形-促音便)
                'VBP': 'め',  # imperative (命令系)
                'VBV': 'もう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'ぬ':
            forms = {
                'VB': 'ぬ',  # plain (終止形)
                'VBI': 'な',  # imperfect (未然形)
                'VBC': 'に',  # conjunctive (連用形)
                'VBCG': 'ん',  # conjunctive geminate (連用形-促音便)
                'VBP': 'ね',  # imperative (命令系)
                'VBV': 'のう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        elif baseform[-1] == 'ぶ':
            forms = {
                'VB': 'ぶ',  # plain (終止形)
                'VBI': 'ば',  # imperfect (未然形)
                'VBC': 'び',  # conjunctive (連用形)
                'VBCG': 'ん',  # conjunctive geminate (連用形-促音便)
                'VBP': 'べ',  # imperative (命令系)
                'VBV': 'ぼう',  # volitional (意志推量形)
                'VBS': ''  # stem/subword token
            }
        else:
            forms = {}
        stem = baseform[:-1]
        return {form: stem + end for form, end in forms.items()}
