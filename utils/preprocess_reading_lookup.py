from collections import defaultdict
import argparse
import json
import xml.etree.ElementTree as ET

import jaconv


def preprocess_reading_lookup(kanjidic_path, output_path):
    """Generate reading to kanji lookup dictionary."""
    reading_lookup = defaultdict(set)
    root = ET.parse(kanjidic_path).getroot()
    characters = root.findall('character')
    print(f'Loaded {len(characters)} characters from kanjidic')
    for c in characters:
        if not c.findtext('misc/grade'):  # only use joyo kanji
            continue
        literal = c.findtext('literal')
        readings = c.findall('reading_meaning/rmgroup/reading')
        for reading in readings:
            if reading.attrib['r_type'] in ['ja_on', 'ja_kun']:
                r = jaconv.hira2kata(reading.text)
                reading_lookup[r].add(literal)

    reading_lookup = {k: list(v) for k, v in reading_lookup.items()}
    with open(output_path, 'w') as f:
        json.dump(reading_lookup, f)
    print(f'Reading lookup output to {output_path}')


def main(args):
    preprocess_reading_lookup(args.kanjidic, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kanjidic',
                        help='Path to KANJIDIC file',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Path to output file',
                        required=True)
    args = parser.parse_args()
    main(args)
