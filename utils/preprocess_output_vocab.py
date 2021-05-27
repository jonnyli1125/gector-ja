import argparse
import os
import bz2


def preprocess_output_vocab(corpora_dir, output_file,
                            edit_tags_file='edit_tagged_sentences.txt.bz2'):
    """Generate output vocab from all corpora."""
    if not os.path.isdir(corpora_dir):
        raise ValueError(f'{corpora_dir} not found')
    edit_vocab = set()
    for root, dirs, files in os.walk(corpora_dir):
        if edit_tags_file in files:
            path = os.path.join(root, edit_tags_file)
            with bz2.open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                tokens = line.decode('utf-8').strip().split()
                for token in tokens:
                    edit = token.rsplit('###')[1]
                    edit_vocab.add(edit)
            print(f'Processed {root}, {len(edit_vocab)} edits')
    lines = [f'{edit}\n' for edit in sorted(edit_vocab) if edit]
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
