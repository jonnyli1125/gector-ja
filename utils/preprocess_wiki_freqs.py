from collections import Counter
import argparse
import os
import re
import json

from fugashi import Tagger


def preprocess_wiki_freqs(source_dir, output_dir,
                          kanji_file='kanji_freq.json',
                          particles_file='particle_freq.json',
                          verbs_file='verbs_freq.json',
                          adjs_file='adjs_freq.json'):
    """Generate frequency lists for kanji and particles for error generation."""
    if not os.path.isdir(source_dir):
        raise ValueError(f'WikiExtractor text folder not found at {source_dir}')
    tagger = Tagger('-Owakati')
    kanji_re = re.compile('([一-龯])')
    kanji_freq = Counter()
    particles_freq = Counter()
    verbs_freq = Counter()
    adjs_freq = Counter()
    for root, dirs, files in os.walk(source_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            with open(fp, encoding='utf-8') as file:
                for line in file.readlines():
                    line = line.strip()
                    if not line or line[0] == '<':  # empty or xml tag
                        continue
                    kanji = kanji_re.findall(line)
                    kanji_freq.update(kanji)
                    tokens = tagger(line)
                    particles_freq.update(t.surface for t in tokens
                                          if t.feature.pos2 == '格助詞')
                    verbs_freq.update(t.feature.orthBase for t in tokens
                                      if t.feature.pos1 == '動詞')
                    adjs_freq.update(t.feature.orthBase for t in tokens
                                     if t.feature.pos1 == '形容詞')
        if not dirs:
            print(f'Finished processing {root}')

    with open(os.path.join(output_dir, kanji_file), 'w') as f:
        json.dump(kanji_freq, f, separators=(',', ':'))
    with open(os.path.join(output_dir, particles_file), 'w') as f:
        json.dump(particles_freq, f, separators=(',', ':'))
    with open(os.path.join(output_dir, verbs_file), 'w') as f:
        json.dump(verbs_freq, f, separators=(',', ':'))
    with open(os.path.join(output_dir, adjs_file), 'w') as f:
        json.dump(adjs_freq, f, separators=(',', ':'))
    print(f'Finished processing Wikipedia dump')
    print(f'Frequency lists output to {output_dir}')


def main(args):
    preprocess_wiki_freqs(args.source_dir, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_dir',
                        help='Path to text folder extracted by WikiExtractor',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to output folder',
                        required=True)
    args = parser.parse_args()
    main(args)
