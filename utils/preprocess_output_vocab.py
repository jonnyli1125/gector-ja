import argparse
import os
import json
from collections import Counter


def preprocess_output_vocab(output_file):
    """Generate output vocab from all corpora."""
    labels = ['[PAD]', '[UNK]']
    with open('data/corpora/jawiki/edit_freq.json') as f:
        jawiki_edit_freq = json.load(f)
    with open('data/corpora/lang8/edit_freq.json') as f:
        lang8_edit_freq = json.load(f)
    edit_freq = Counter(jawiki_edit_freq)
    edit_freq.update(lang8_edit_freq)
    ordered = sorted(edit_freq.items(), key=lambda x: x[1], reverse=True)
    labels += [edit for edit, freq in ordered if freq >= 5000]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(f'{label}\n' for label in labels)
    print(f'{len(labels)} edits output to {output_file}.')


def main(args):
    preprocess_output_vocab(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        help='Path to output file',
                        required=True)
    args = parser.parse_args()
    main(args)
