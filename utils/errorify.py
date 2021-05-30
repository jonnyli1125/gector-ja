import re
import json
import traceback

import numpy as np
from numpy.random import choice, randint, uniform
from fugashi import Tagger


class Errorify:
    """Generate artificial errors in sentences."""

    def __init__(self, kanji2freq_path='./data/kanji_freq.json',
                 particle2freq_path='./data/particle_freq.json',
                 reading2kanji_path='./data/reading_lookup.json',
                 transitivity_pairs_path='./data/transitivity_pairs.json'):
        self.tagger = Tagger('-Owakati')
        self.num_errors_prob = [0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.15,
                                0.15, 0.13]
        self.error_funcs_prob = [.20, .30, .25, .25]
        self.error_funcs = [
            self.delete_error,
            self.inflection_error,
            self.insert_error,
            self.replace_error
        ]
        with open(kanji2freq_path, 'r') as f:
            kanji2freq = json.load(f)
            self.kanji = list(kanji2freq.keys())
            kanji_freq = np.array(list(kanji2freq.values()))
            self.kanji_prob = kanji_freq / kanji_freq.sum()
        with open(particle2freq_path, 'r') as f:
            particle2freq = json.load(f)
            self.particles = list(particle2freq.keys())
            particles_freq = np.array(list(particle2freq.values()))
            self.particles_prob = particles_freq / particles_freq.sum()
        with open(reading2kanji_path, 'r') as f:
            self.reading_lookup = json.load(f)
        self.kanji_re = re.compile('([一-龯])')
        self.typo_chars = ['っ', '。', '、', '.', ',']

    def delete_error(self, sentence):  # delete copula
        """Delete a random token or character."""
        tokens = self.tagger(sentence)
        surface_tokens = [w.surface for w in tokens]
        candidate_tokens = [i for i, t in enumerate(tokens)
                            if '記号' not in t.feature.pos1]
        if not candidate_tokens:
            return sentence
        i = choice(candidate_tokens)  # pick a token
        j = randint(len(surface_tokens[i])+1)  # pick a char or whole token
        if j == len(surface_tokens[i]):
            del surface_tokens[i]
        else:
            surface_tokens[i] = surface_tokens[i][:j] + surface_tokens[i][j+1:]
        return ''.join(surface_tokens)

    def inflection_error(self, sentence):
        """Misinflect a random verb/adj stem."""
        tokens = self.tagger(sentence)
        surface_tokens = [w.surface for w in tokens]
        candidate_tokens = [i for i, t in enumerate(tokens)
                            if t.feature.pos1 in ['動詞', '形容詞']]
        if not candidate_tokens:
            return sentence
        i = choice(candidate_tokens)
        baseform = tokens[i].feature.orthBase or tokens[i].feature.lemma
        if not baseform:
            return sentence
        morphs = list(self.get_forms(baseform).values())
        if not morphs:
            return sentence
        surface_tokens[i] = choice(morphs)
        return ''.join(surface_tokens)

    def insert_error(self, sentence):
        """Insert a random kanji, particle, or typo character."""
        tokens = [w.surface for w in self.tagger(sentence)]
        if not tokens:
            return sentence
        i = randint(len(tokens))
        rand = uniform()
        if rand < 0.1:  # pick a kanji
            new_word = choice(self.kanji, p=self.kanji_prob)
        elif rand < 0.2:  # pick a typo character
            new_word = choice(self.typo_chars)
        else:  # pick a particle
            new_word = choice(self.particles, p=self.particles_prob)
        tokens[i] += new_word
        return ''.join(tokens)

    def replace_error(self, sentence):
        """Replace a random particle or kanji/verb/adj with same reading."""
        tokens = self.tagger(sentence)
        surface_tokens = [w.surface for w in tokens]
        candidate_tokens = [i for i, t in enumerate(tokens)
                            if t.feature.pos2 == '格助詞'
                            or self.kanji_re.search(t.surface)]
        if not candidate_tokens:
            return sentence
        i = choice(candidate_tokens)
        token = tokens[i]
        if token.feature.pos2 == '格助詞':
            repl_word = choice(self.particles, p=self.particles_prob)
        elif token.feature.pos1 in ['動詞', '形容詞']:
            kanaBase = token.feature.kanaBase
            if not kanaBase:
                return sentence
            reading = f'{kanaBase[:-1]}.{kanaBase[-1]}'
            if reading not in self.reading_lookup:
                return sentence
            ending = token.surface[len(token.feature.orthBase)-1:]
            repl_word = choice(self.reading_lookup[reading]) + ending
        else:
            kanji = self.kanji_re.findall(token.surface)
            if not kanji:
                return sentence
            j = randint(len(kanji))
            reading = self.tagger(kanji[j])[0].feature.kana
            if reading not in self.reading_lookup:
                return sentence
            repl_kanji = choice(self.reading_lookup[reading])
            repl_word = token.surface.replace(kanji[j], repl_kanji, 1)
        surface_tokens[i] = repl_word
        return ''.join(surface_tokens)

    def __call__(self, sentence):
        """Get sentence with artificially generated errors."""
        error_sent = sentence
        count = choice(range(len(self.num_errors_prob)), p=self.num_errors_prob)
        for i in range(count):
            try:
                error_func = choice(self.error_funcs, p=self.error_funcs_prob)
                error_sent = error_func(error_sent)
            except Exception:
                print(sentence)
                traceback.print_exc()
        return error_sent

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
