from collections import Counter
import argparse
import os
import json
import re

from fugashi import Tagger


invalid_bytes_re = re.compile(r'[\x00-\x1F]+')
sline_re = re.compile(r'\[sline\].*?\[/sline\]')
color_tags = ['[f-blue]','[/f-blue]',
              '[f-red]','[/f-red]',
              '[f-bold]','[/f-bold]']


def clean_line(line):
    line = line.strip()
    for tag in color_tags:
        line = line.replace(tag, '')
    line = sline_re.sub('', line).replace('[/sline]', '')
    return line


def preprocess_lang8_freqs(source_file, output_dir,
                           verbs_file='verbs_freq.json',
                           adjs_file='adjs_freq.json'):
    """Generate frequency dicts from Lang8 corpus."""
    verbs_freq = {}
    adjs_freq = {}
    with open(os.path.join(output_dir, verbs_file)) as f:
        verbs_freq = Counter(json.load(f))
    with open(os.path.join(output_dir, adjs_file)) as f:
        adjs_freq = Counter(json.load(f))
    tagger = Tagger('-Owakati')
    lines = []
    with open(source_file, encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        row = json.loads(invalid_bytes_re.sub('', line))
        if row[2] != 'Japanese':
            continue
        for corrections in row[5]:
            for correction in corrections:
                if not correction:
                    continue
                correction = clean_line(correction)
                tokens_corr = tagger(correction)
                verbs_freq.update(t.feature.orthBase for t in tokens_corr
                                  if t.feature.pos1 == '動詞')
                adjs_freq.update(t.feature.orthBase for t in tokens_corr
                                 if t.feature.pos1 == '形容詞')
        if i % 100 == 0:
            print(f'{i}/{len(lines)}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, verbs_file), 'w') as f:
        json.dump(verbs_freq, f)
    with open(os.path.join(output_dir, adjs_file), 'w') as f:
        json.dump(adjs_freq, f)
    print('Verb/adj frequencies updated on Lang8 corpus.')

def main(args):
    preprocess_lang8_freqs(args.source, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Path to Lang8 corpus file',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to output directory',
                        required=True)
    args = parser.parse_args()
    main(args)
