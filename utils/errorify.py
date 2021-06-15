import re
import json
import traceback

import numpy as np
from numpy.random import choice, randint, uniform, binomial
from fugashi import Tagger


class Errorify:
    """Generate artificial errors in sentences."""

    def __init__(self, reading_lookup_path='./data/reading_lookup.json'):
        self.tagger = Tagger('-Owakati')
        self.error_prob = [0.05, 0.07, 0.25, 0.35, 0.28]
        self.core_particles = ['が', 'の', 'を', 'に', 'へ', 'と', 'で', 'から',
                               'より', 'は', 'も']
        self.other_particles = ['か', 'の', 'や', 'に', 'と', 'やら', 'なり',
                                'だの', 'ばかり', 'まで', 'だけ', 'ほど', 'くらい',
                                'など', 'やら', 'こそ', 'でも', 'しか', 'さえ',
                                'だに', 'ば', 'て', 'のに', 'ので', 'から']
        with open(reading_lookup_path) as f:
            self.reading_lookup = json.load(f)

    def delete_error(self, token, feature):
        """Delete a token."""
        return ''

    def inflection_error(self, token, feature):
        """Misinflect a verb/adj stem."""
        baseform = feature.orthBase or feature.lemma
        if not baseform:
            return token
        morphs = list(self.get_forms(baseform).values())
        if not morphs:
            return token
        return choice(morphs)

    def insert_error(self, token, feature):
        """Insert a random particle."""
        return token + choice(self.other_particles)

    def replace_error(self, token, feature):
        """Replace a particle or word with another word of the same reading."""
        if feature.pos2 in ['格助詞', '係助詞']:
            return choice(self.core_particles)
        elif feature.pos1 in ['動詞', '形容詞']:
            reading = f'{feature.kanaBase[:-1]}.{feature.kanaBase[-1]}'
            if reading not in self.reading_lookup:
                return token
            ending = token[len(feature.orthBase)-1:]
            return choice(self.reading_lookup[reading]) + ending
        else:
            if feature.kanaBase not in self.reading_lookup:
                return token
            return choice(self.reading_lookup[feature.kanaBase])

    def __call__(self, sentence):
        """Get sentence with artificially generated errors."""
        # need to this because fugashi has some weird bug
        tokens = [(t.surface, t.feature) for t in self.tagger(sentence)]
        tokens_surface = [t[0] for t in tokens]
        n_errors = choice(range(len(self.error_prob)), p=self.error_prob)
        candidate_tokens = [i for i, (t, f) in enumerate(tokens)
                            if f.pos2 not in ['数詞', '固有名詞']
                            and f.pos1 not in ['記号', '補助記号']]
        if not candidate_tokens:
            return sentence
        error_token_ids = choice(candidate_tokens, size=(n_errors,))
        for token_id in error_token_ids:
            token, feat = tokens[token_id]
            if feat.pos2 in ['格助詞', '係助詞']:
                error_func = choice([self.delete_error, self.replace_error])
            elif feat.pos1 in ['動詞', '形容詞']:
                error_func = choice([self.replace_error, self.inflection_error],
                                    p=[0.05, 0.95])
            elif feat.pos1 == '名詞':
                error_func = choice([self.insert_error, self.replace_error],
                                    p=[0.05, 0.95])
            else:
                error_func = choice([self.insert_error, self.delete_error],
                                    p=[0.05, 0.95])
            tokens_surface[token_id] = error_func(token, feat)
        return ''.join(tokens_surface)

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
