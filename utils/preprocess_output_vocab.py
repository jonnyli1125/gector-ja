import argparse
import os
import json
from collections import Counter


def preprocess_output_vocab(output_file, weights_file):
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
    n_samples = sum(edit_freq[edit] for edit in labels)
    n_classes = len(labels)
    class_weights = [1.] * n_classes
    for i, edit in enumerate(labels):
        if edit in edit_freq:
            class_weights[i] = n_samples / (n_classes * edit_freq[edit])
    class_weights_d = [
        1.,
        n_samples / (3 * edit_freq['$KEEP']),
        n_samples / (3 * sum(edit_freq[edit] for edit in labels[3:]))
    ]
    with open(weights_file, 'w', encoding='utf-8') as f:
        json.dump([class_weights, class_weights_d], f)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(f'{label}\n' for label in labels)
    print(f'{n_classes} edits output to {output_file}.')
    print(f'Class weights output to {weights_file}.')


def main(args):
    preprocess_output_vocab(args.output, args.weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        help='Path to output file',
                        required=True)
    parser.add_argument('-w', '--weights',
                        help='Path to class weights output file',
                        required=True)
    args = parser.parse_args()
    main(args)
