import argparse
import os
import json

from transformers import AutoTokenizer


def preprocess_output_vocab(output_file):
    """Generate output vocab from all corpora."""
    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
    labels = ['[PAD]', '[UNK]', '$KEEP', '$DELETE']
    vb_transform = ['VB', 'VBI', 'VBC', 'VBCG', 'VBP', 'VBV', 'VBS']
    adj_transform = ['ADJ', 'ADJC', 'ADJCG', 'ADJS']
    labels.extend(f'$TRANSFORM_{f1}_{f2}'
                  for f1 in vb_transform for f2 in vb_transform if f1 != f2)
    labels.extend(f'$TRANSFORM_{f1}_{f2}'
                  for f1 in adj_transform for f2 in adj_transform if f1 != f2)
    with open('data/wordpiece_freq.json') as f:
        wps_freq = json.load(f)
    freqs = sorted(wps_freq.items(), key=lambda x: x[1])
    total = sum(f[1] for f in freqs)
    dist = [f[1] / total for f in freqs]
    m = len(freqs) // 4
    print(sum(dist[-m:]))
    filtered_vocab = sorted(word for word, freq in freqs[-m:])
    labels.extend(f'$APPEND_{word}' for word in filtered_vocab)
    labels.extend(f'$REPLACE_{word}' for word in filtered_vocab)
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
