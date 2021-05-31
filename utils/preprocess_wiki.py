import argparse
import os
import re
import unicodedata
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
    root, fn, output_dir = args
    corr_lines = []
    incorr_lines = []
    edit_rows = []
    fp = os.path.join(root, fn)
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
                levels = edit_tagger(error_sent, sent)
                edit_rows.extend(levels)
    base = os.path.basename(root)
    base_path = os.path.join(output_dir, base, fn)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    corr_path = os.path.join(base_path, correct_file)
    incorr_path = os.path.join(base_path, incorrect_file)
    edit_tags_path = os.path.join(base_path, edit_tags_file)
    with open(corr_path, 'w', encoding='utf-8') as file:
        file.writelines(corr_lines)
    with open(incorr_path, 'w', encoding='utf-8') as file:
        file.writelines(incorr_lines)
    write_dataset(edit_tags_path, edit_rows)
    print(f'Processed {len(corr_lines)} sentences, ' \
          f'{len(edit_rows)} edit-tagged sentences in {fp}')
    return len(corr_lines), len(edit_rows)


def preprocess_wiki(source_dir, output_dir, processes):
    """Generate synthetic error corpus from Wikipedia dump."""
    if not os.path.isdir(source_dir):
        raise ValueError(f'WikiExtractor text folder not found at {source_dir}')
    n_sents = 0
    n_edit_sents = 0
    pool_inputs = []
    for root, dirs, files in os.walk(source_dir):
        if not dirs:
            for fn in files:
                pool_inputs.append([root, fn, output_dir])
    print(f'Processing {len(pool_inputs)} parts...')
    pool = Pool(processes)
    pool_outputs = pool.imap_unordered(preprocess_wiki_part, pool_inputs)
    n_sents = 0
    n_edit_sents = 0
    for n in pool_outputs:
        n_sents += n[0]
        n_edit_sents += n[1]
    print(f'Processed {n_sents} sentences, {n_edit_sents} edit-tagged ' \
          'sentences from Wikipedia dump')
    print(f'Synthetic error corpus output to {output_dir}')


def main(args):
    preprocess_wiki(args.source_dir, args.output_dir, args.processes)


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
    args = parser.parse_args()
    main(args)
