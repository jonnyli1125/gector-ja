import argparse
import os
import re
import unicodedata
import json
from collections import Counter
from multiprocessing import Pool

from .errorify import Errorify
from .edits import EditTagger
from .helpers import write_dataset


en_sentence_re = re.compile(r'([a-zA-Z]+[\W]*\s){5,}')
errorify = Errorify()
edit_tagger = EditTagger()


def preprocess_wiki_part(args,
                         correct_file='corr_sentences.txt',
                         incorrect_file='incorr_sentences.txt',
                         edit_tags_file='edit_tagged_sentences.tfrec.gz'):
    edit_tagger.edit_freq = Counter()
    root, fn, output_dir, use_existing = args
    edit_rows = []
    fp = os.path.join(root, fn)
    base = os.path.basename(root)
    base_path = os.path.join(output_dir, base, fn)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    corr_path = os.path.join(base_path, correct_file)
    incorr_path = os.path.join(base_path, incorrect_file)
    if use_existing:
        with open(corr_path, 'r', encoding='utf-8') as f:
            corr_lines = f.readlines()
        with open(incorr_path, 'r', encoding='utf-8') as f:
            incorr_lines = f.readlines()
    else:
        corr_lines = []
        incorr_lines = []
        with open(fp, encoding='utf-8') as file:
            skip = False
            for line in file.readlines():
                line = line.strip()
                if not line or line[0] == '<' or line[-1] == '.' or skip:
                    skip = False
                    continue
                if line[-1] != '。':
                    skip = True
                    continue
                if en_sentence_re.search(line):
                    continue
                line = unicodedata.normalize('NFKC', line).replace(' ', '')
                quote_lvl = 0
                brackets_lvl = 0
                start_i = 0
                sents = []
                for i, c in enumerate(line):
                    if c == '「':
                        quote_lvl += 1
                    elif c == '」':
                        quote_lvl -= 1
                    elif c == '(':
                        brackets_lvl += 1
                    elif c == ')':
                        brackets_lvl -= 1
                    elif c == '。' and quote_lvl == 0 and brackets_lvl == 0:
                        sents.append(line[start_i:i+1])
                        start_i = i+1
                for sent in sents:
                    sent = sent.strip().lstrip('。')
                    if not sent:
                        continue
                    error_sent = errorify(sent)
                    corr_lines.append(f'{sent}\n')
                    incorr_lines.append(f'{error_sent}\n')
        with open(corr_path, 'w', encoding='utf-8') as file:
            file.writelines(corr_lines)
        with open(incorr_path, 'w', encoding='utf-8') as file:
            file.writelines(incorr_lines)
    for incorr_line, corr_line in zip(incorr_lines, corr_lines):
        incorr_line = incorr_line.strip()
        corr_line = corr_line.strip()
        if not incorr_line or not corr_line:
            continue
        levels = edit_tagger(incorr_line, corr_line)
        edit_rows.extend(levels)
    edit_tags_path = os.path.join(base_path, edit_tags_file)
    write_dataset(edit_tags_path, edit_rows)
    print(f'Processed {len(corr_lines)} sentences, ' \
          f'{len(edit_rows)} edit-tagged sentences in {fp}')
    return len(corr_lines), len(edit_rows), edit_tagger.edit_freq


def preprocess_wiki(source_dir, output_dir, processes, use_existing,
                    edit_freq_file='edit_freq.json'):
    """Generate synthetic error corpus from Wikipedia dump."""
    if not os.path.isdir(source_dir):
        raise ValueError(f'WikiExtractor text folder not found at {source_dir}')
    n_sents = 0
    n_edit_sents = 0
    pool_inputs = []
    for root, dirs, files in os.walk(source_dir):
        if not dirs:
            for fn in files:
                pool_inputs.append([root, fn, output_dir, use_existing])
    print(f'Processing {len(pool_inputs)} parts...')
    pool = Pool(processes)
    pool_outputs = pool.imap_unordered(preprocess_wiki_part, pool_inputs)
    n_sents = 0
    n_edit_sents = 0
    edit_freq = Counter()
    for n in pool_outputs:
        n_sents += n[0]
        n_edit_sents += n[1]
        edit_freq.update(n[2])

    with open(os.path.join(output_dir, edit_freq_file), 'w') as f:
        json.dump(edit_freq, f)
    print(f'Processed {n_sents} sentences, {n_edit_sents} edit-tagged ' \
          'sentences from Wikipedia dump')
    print(f'Synthetic error corpus output to {output_dir}')


def main(args):
    preprocess_wiki(args.source_dir, args.output_dir, args.processes,
                    args.use_existing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_dir',
                        help='Path to text folder extracted by WikiExtractor',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to output folder',
                        required=True)
    parser.add_argument('-p', '--processes', type=int,
                        help='Number of processes',
                        required=False)
    parser.add_argument('-e', '--use_existing',
                        help='Edit tag existing error-generated sentences',
                        action='store_true')
    args = parser.parse_args()
    main(args)
