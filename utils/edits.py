from difflib import SequenceMatcher

from transformers import AutoTokenizer
import numpy as np
import Levenshtein


class EditTagger:
    """
    Get edit sequences to transform source sentence to target sentence.

    Original reference code @ https://github.com/grammarly/gector (see README).
    """

    def __init__(self, verb_adj_forms_path='data/transform.txt'):
        self.tokenizer = AutoTokenizer.from_pretrained(
            'cl-tohoku/bert-base-japanese-v2')
        encode, decode = self.get_verb_adj_form_dicts(verb_adj_forms_path)
        self.encode_verb_adj_form = encode
        self.decode_verb_adj_form = decode
        self.op_delim = '###'
        self.token_delim = ' '

    def get_verb_adj_form_dicts(self, verb_adj_forms_path):
        encode, decode = {}, {}
        with open(verb_adj_forms_path, 'r', encoding='utf-8') as f:
            for line in f:
                words, tags = line.split(':')
                tags = tags.strip()
                word1, word2 = words.split('_')
                tag1, tag2 = tags.split('_')
                decode_key = f'{word1}_{tag1}_{tag2}'
                if decode_key not in decode:
                    encode[words] = tags
                    decode[decode_key] = word2
        return encode, decode

    def tokenize(self, sentence, **kwargs):
        ids = self.tokenizer(sentence, **kwargs)['input_ids']
        return self.tokenizer.convert_ids_to_tokens(ids)

    def join_tokens(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens).replace(' ', '')

    def __call__(self, source, target):
        edit_lines = []
        edit_levels = self.get_edit_levels(source, target)
        # edit_levels = [self.get_edits(source, target)]
        for cur_tokens, cur_edits in edit_levels:
            labelled_tokens = []
            for token, edit_list in zip(cur_tokens, cur_edits):
                edit = edit_list[0]
                labelled_tokens.append(f'{token}{self.op_delim}{edit}')
            edit_line = self.token_delim.join(labelled_tokens)
            edit_lines.append(edit_line)
        return edit_lines

    def get_edits(self, source, target, add_special_tokens=True):
        source_tokens = self.tokenize(source,
                                      add_special_tokens=add_special_tokens)
        target_tokens = self.tokenize(target, add_special_tokens=True)
        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        diffs = list(matcher.get_opcodes())
        edits = []
        for tag, i1, i2, j1, j2 in diffs:
            source_part = source_tokens[i1:i2]
            target_part = target_tokens[j1:j2]
            if tag == 'equal':
                continue
            elif tag == 'delete':
                for i in range(i1, i2):
                    edits.append((i, '$DELETE'))
            elif tag == 'insert':
                for target_token in target_part:
                    edits.append((i1-1, f'$APPEND_{target_token}'))
            else:  # tag == 'replace'
                _, alignments = self.perfect_align(source_part, target_part)
                for alignment in alignments:
                    new_edits = self.convert_alignment_into_edits(alignment, i1)
                    edits.extend(new_edits)

        # map edits to source tokens
        labels = [['$KEEP'] for i in range(len(source_tokens))]
        for i, edit in edits:
            if labels[i] == ['$KEEP']:
                labels[i] = []
            labels[i].append(edit)

        return source_tokens, labels

    def perfect_align(self, t, T, insertions_allowed=0,
                      cost_function=Levenshtein.distance):
        # dp[i, j, k] is a minimal cost of matching first `i` tokens of `t` with
        # first `j` tokens of `T`, after making `k` insertions after last match
        # of token from `t`. In other words t[:i] aligned with T[:j].

        # Initialize with INFINITY (unknown)
        shape = (len(t) + 1, len(T) + 1, insertions_allowed + 1)
        dp = np.ones(shape, dtype=int) * int(1e9)
        come_from = np.ones(shape, dtype=int) * int(1e9)
        come_from_ins = np.ones(shape, dtype=int) * int(1e9)

        dp[0, 0, 0] = 0  # Starting point. Nothing matched to nothing.
        for i in range(len(t) + 1):  # Go inclusive
            for j in range(len(T) + 1):  # Go inclusive
                for q in range(insertions_allowed + 1):  # Go inclusive
                    if i < len(t):
                        # Given matched sequence of t[:i] and T[:j], match token
                        # t[i] with following tokens T[j:k].
                        for k in range(j, len(T) + 1):
                            T_jk = '   '.join(T[j:k])
                            transform = self.get_g_trans(t[i], T_jk)
                            if transform:
                                cost = 0
                            else:
                                cost = cost_function(t[i], T_jk)
                            current = dp[i, j, q] + cost
                            if dp[i + 1, k, 0] > current:
                                dp[i + 1, k, 0] = current
                                come_from[i + 1, k, 0] = j
                                come_from_ins[i + 1, k, 0] = q
                    if q < insertions_allowed:
                        # Given matched sequence of t[:i] and T[:j], create
                        # insertion with following tokens T[j:k].
                        for k in range(j, len(T) + 1):
                            cost = len('   '.join(T[j:k]))
                            current = dp[i, j, q] + cost
                            if dp[i, k, q + 1] > current:
                                dp[i, k, q + 1] = current
                                come_from[i, k, q + 1] = j
                                come_from_ins[i, k, q + 1] = q

        # Solution is in the dp[len(t), len(T), *]. Backtracking from there.
        alignment = []
        i = len(t)
        j = len(T)
        q = dp[i, j, :].argmin()
        while i > 0 or q > 0:
            is_insert = (come_from_ins[i, j, q] != q) and (q != 0)
            j, k, q = come_from[i, j, q], j, come_from_ins[i, j, q]
            if not is_insert:
                i -= 1

            if is_insert:
                alignment.append(['INSERT', T[j:k], i])
            else:
                alignment.append([f'REPLACE_{t[i]}', T[j:k], i])

        assert j == 0

        return dp[len(t), len(T)].min(), list(reversed(alignment))

    def get_g_trans(self, source_token, target_token):
        # check equal
        if source_token == target_token:
            return '$KEEP'
        # check transform verb/adj form possible
        key = f'{source_token}_{target_token}'
        encoding = self.encode_verb_adj_form.get(key, '')
        if source_token and encoding:
            return f'$TRANSFORM_{encoding}'
        return None

    def convert_alignment_into_edits(self, alignment, i1):
        edits = []
        action, target_tokens, new_idx = alignment
        shift_idx = new_idx + i1
        source_token = action.replace('REPLACE_', '')

        # check if delete
        if not target_tokens:
            return [(shift_idx, '$DELETE')]

        # check splits
        for i in range(1, len(target_tokens)):
            target_token = ''.join(target_tokens[:i + 1])
            transform = self.get_g_trans(source_token, target_token)
            if transform:
                edits.append((shift_idx, transform))
                for target in target_tokens[i + 1:]:
                    edits.append((shift_idx, f'$APPEND_{target}'))
                return edits

        # default case
        transform_costs = []
        transforms = []
        for target_token in target_tokens:
            transform = self.get_g_trans(source_token, target_token)
            if transform:
                cost = 0
            else:
                cost = Levenshtein.distance(source_token, target_token)
            transforms.append(transform)
            transform_costs.append(cost)
        min_cost_idx = np.argmin(transform_costs)
        # append everything before min cost token (target) to the previous word
        for i in range(min_cost_idx):
            edits.append((shift_idx - 1, f'$APPEND_{target_tokens[i]}'))
        # replace/transform target word
        transform = transforms[min_cost_idx]
        if transform:
            target = transform
        else:
            target = f'$REPLACE_{target_tokens[min_cost_idx]}'
        edits.append((shift_idx, target))
        # append everything after target to this word
        for i in range(min_cost_idx + 1, len(target_tokens)):
            edits.append((shift_idx, f'$APPEND_{target_tokens[i]}'))
        return edits

    def get_edit_levels(self, source, target, max_iter=10):
        levels = []
        cur_sent = source
        for i in range(max_iter):
            cur_tokens, cur_edits = self.get_edits(cur_sent, target,
                                                   add_special_tokens=(i==0))
            if i > 0 and all(e == ['$KEEP'] for e in cur_edits):
                break
            levels.append((cur_tokens, cur_edits))
            new_tokens = self.apply_edits(cur_tokens, cur_edits)
            cur_sent = self.join_tokens(new_tokens)
        # tokenizer may produce [UNK] so we can't actually assert this
        # assert cur_sent == target
        return levels

    def apply_edits(self, source_tokens, edits):
        new_tokens = []
        for i, (token, edit_list) in enumerate(zip(source_tokens, edits)):
            edit = edit_list[0]
            if edit == '$KEEP':
                new_tokens.append(token)
            elif edit == '$DELETE':
                continue
            elif edit.startswith('$APPEND_'):
                new_tokens += [token, edit.replace('$APPEND_', '')]
            elif edit.startswith('$REPLACE_'):
                new_tokens.append(edit.replace('$REPLACE_', ''))
            elif edit.startswith('$TRANSFORM_'):
                transform = edit.replace('$TRANSFORM_', '')
                decode_key = f'{token}_{transform}'
                new_tokens.append(self.decode_verb_adj_form[decode_key])
            else:
                raise ValueError(f'Invalid edit {edit}')
        return new_tokens
