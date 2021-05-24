import argparse
import json
import re
import unicodedata
import bz2

import langid

from edits import EditTagger


invalid_bytes_re = re.compile(r'[\x00-\x1F]+')
sline_re = re.compile(r'\[sline\].*?\[/sline\]')
color_tags = ['[f-blue]','[/f-blue]',
              '[f-red]','[/f-red]',
              '[f-bold]','[/f-bold]']


def clean_line(line):
    line = line.strip()
    if line.endswith('GOOD'):
        line = line[:-4]
    elif line.endswith('OK'):
        line = line[:-2]
    line = line.strip()
    for tag in color_tags:
        line = line.replace(tag, '')
    line = sline_re.sub('', line).replace('[/sline]', '')
    return unicodedata.normalize('NFKC', line).replace(' ', '')


def preprocess_lang8(source_file, output_file):
    """Generate edit-tagged sentence corpus from Lang8 corpus."""
    edit_tagger = EditTagger()
    lines = []
    with open(source_file) as f:
        lines = f.readlines()
    edit_lines = []
    for line in lines:
        row = json.loads(invalid_bytes_re.sub('', line))
        if row[2] != 'Japanese':
            continue
        for learner_sent, corrections in zip(row[4], row[5]):
            learner_sent = clean_line(learner_sent)
            if langid.classify(learner_sent)[0] != 'ja':
                continue
            edit_tagged_lines = []
            target_sent = learner_sent
            for correction in corrections:
                correction = clean_line(correction)
                if not correction or langid.classify(correction) != 'ja':
                    continue
                edit_tagged_lines.extend(edit_tagger(learner_sent, target_sent))
            for edit_tagged_line in edit_tagged_lines:
                edit_lines.append(f'{edit_tagged_line}\n')
    with open(output_file, 'wb') as file:
        edit_lines_bytes = ''.join(edit_lines).encode('utf-8')
        edit_lines_compressed = bz2.compress(edit_lines_bytes)
        file.write(edit_lines_compressed)


def main(args):
    preprocess_lang8(args.source, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source',
                        help='Path to Lang8 corpus file',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Path to output file',
                        required=True)
    args = parser.parse_args()
    main(args)
