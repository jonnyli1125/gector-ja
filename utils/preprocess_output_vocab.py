import argparse
import os

import tensorflow as tf

from helpers import read_dataset


def preprocess_output_vocab(corpora_dir, output_file,
                            edit_tags_file='edit_tagged_sentences.tfrec.gz'):
    """Generate output vocab from all corpora."""
    if not os.path.isdir(corpora_dir):
        raise ValueError(f'{corpora_dir} not found')
    edit_vocab = set()
    paths = []
    for root, dirs, files in os.walk(corpora_dir):
        if edit_tags_file in files:
            path = os.path.join(root, edit_tags_file)
            paths.append(path)
    dataset = read_dataset(paths)
    print(dataset)
    print(f'Loaded dataset')
    for i, example in enumerate(dataset):
        edits = example['labels'].numpy()
        edit_vocab.update(e.decode() for e in edits)
        if i % 10000 == 0:
            print(f'{i} processed, {len(edit_vocab)} edits')
    edit_vocab -= {'$KEEP', '$DELETE', ''}
    lines = ['$KEEP\n', '$DELETE\n']
    lines += [f'{edit}\n' for edit in sorted(edit_vocab)]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'{len(edit_vocab)} edits output to {output_file}.')


def main(args):
    preprocess_output_vocab(args.corpora_dir, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpora_dir',
                        help='Path to corpora directory',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Path to output file',
                        required=True)
    args = parser.parse_args()
    main(args)
