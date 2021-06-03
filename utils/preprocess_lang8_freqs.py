from collections import Counter
import argparse
import os
import json
import re
from multiprocessing import Pool

from fugashi import Tagger
from transformers import AutoTokenizer


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


tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')


def preprocess_lang8_part(rows):
    wps_freq = Counter()
    for row in rows:
        for corrections in row[5]:
            for target_sent in corrections:
                if not target_sent:
                    continue
                target_sent = clean_line(target_sent)
                #tokens_corr = tagger(correction)
                #verbs_freq.update(t.feature.orthBase for t in tokens_corr
                #                  if t.feature.pos1 == '動詞')
                #adjs_freq.update(t.feature.orthBase for t in tokens_corr
                #                 if t.feature.pos1 == '形容詞')
                wp_ids = tokenizer(target_sent, add_special_tokens=False,
                    return_tensors='np')['input_ids'][0]
                wps = tokenizer.convert_ids_to_tokens(wp_ids)
                wps_freq.update(wps)
    return wps_freq


def preprocess_lang8_freqs(source_file, output_dir, processes,
                           wps_file='wordpiece_freq.json'):
    """Generate edit-tagged sentence corpus from Lang8 corpus."""
    lines = []
    with open(source_file, encoding='utf-8') as f:
        lines = f.readlines()
    rows = []
    for line in lines:
        row = json.loads(invalid_bytes_re.sub('', line))
        if row[2] == 'Japanese':
            rows.append(row)
    r = 512
    rows_parts = [rows[i:i + r] for i in range(0, len(rows), r)]
    print(f'Loaded {len(rows)} Japanese entries into {len(rows_parts)} parts')
    pool = Pool(processes)
    pool_outputs = pool.imap_unordered(preprocess_lang8_part, rows_parts)
    #verbs_freq = {}
    #adjs_freq = {}
    wps_freq = {}
    #with open(os.path.join(output_dir, verbs_file)) as f:
    #    verbs_freq = Counter(json.load(f))
    #with open(os.path.join(output_dir, adjs_file)) as f:
    #    adjs_freq = Counter(json.load(f))
    with open(os.path.join(output_dir, wps_file)) as f:
        wps_freq = Counter(json.load(f))
    for dct in pool_outputs:
        wps_freq.update(dct)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #with open(os.path.join(output_dir, verbs_file), 'w') as f:
    #    json.dump(verbs_freq, f)
    #with open(os.path.join(output_dir, adjs_file), 'w') as f:
    #    json.dump(adjs_freq, f)
    with open(os.path.join(output_dir, wps_file), 'w') as f:
        json.dump(wps_freq, f)
    print('Verb/adj frequencies updated on Lang8 corpus.')


def main(args):
    preprocess_lang8_freqs(args.source, args.output_dir, args.processes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Path to Lang8 corpus file',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Path to output directory',
                        required=True)
    parser.add_argument('-p', '--processes', type=int,
                        help='Number of processes',
                        required=False)
    args = parser.parse_args()
    main(args)
