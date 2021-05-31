import argparse
import json
from itertools import chain

from fugashi import Tagger

from .errorify import Errorify


def preprocess_transformations(verbs_file, adjs_file, output_file):
    """Generate verb/adj stem transformations for all verbs in vocab."""

    with open(verbs_file) as f:
        verbs = list(json.load(f).keys())
    with open(adjs_file) as f:
        adjs = list(json.load(f).keys())
    print(f'Loaded {len(verbs)} verbs and {len(adjs)} adjectives.')
    errorify = Errorify()
    lines = []
    for baseform in chain(verbs, adjs):
        forms = errorify.get_forms(baseform)
        for form1, orth1 in forms.items():
            for form2, orth2 in forms.items():
                if form1 != form2:
                    lines.append(f'{orth1}_{orth2}:{form1}_{form2}\n')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'Wrote {len(lines)} transformations to {output_file}.')

def main(args):
    preprocess_transformations(args.verbs, args.adjs, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbs',
                        help='Path to verbs frequencies file',
                        required=True)
    parser.add_argument('-a', '--adjs',
                        help='Path to i-adjectives frequencies file',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Path to output file',
                        required=True)
    args = parser.parse_args()
    main(args)
