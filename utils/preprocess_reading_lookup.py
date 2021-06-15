from collections import defaultdict
import argparse
import json
import xml.etree.ElementTree as ET
import re

import jaconv


kanji_re = re.compile(r'([一-龯])')


def preprocess_reading_lookup(kanjidic_path, jmdict_path, output_path):
    """Generate reading to kanji lookup dictionary."""
    reading_lookup = defaultdict(set)
    kd_root = ET.parse(kanjidic_path).getroot()
    characters = kd_root.findall('character')
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
    jd_root = ET.parse(jmdict_path).getroot()
    entries = jd_root.findall('entry')
    print(f'Loaded {len(entries)} entries from JMdict')
    for e in entries:
        pos = e.findtext('sense/pos')
        pri = e.findtext('k_ele/ke_pri')
        if pri and 'noun' in pos and kanji_re.search(e.findtext('k_ele/keb')):
            reading = {jaconv.hira2kata(r.text) for r in e.findall('r_ele/reb')}
            orth = {k.text for k in e.findall('k_ele/keb')}
            for r in reading:
                reading_lookup[r] |= orth
    reading_lookup = {k: list(v) for k, v in reading_lookup.items()}
    with open(output_path, 'w') as f:
        json.dump(reading_lookup, f)
    print(f'Reading lookup output to {output_path}')


def main(args):
    preprocess_reading_lookup(args.kanjidic, args.jmdict, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kanjidic',
                        help='Path to KANJIDIC file',
                        required=True)
    parser.add_argument('-j', '--jmdict',
                        help='Path to JMDict file',
                        required=True)
    parser.add_argument('-o', '--output',
                        help='Path to output file',
                        required=True)
    args = parser.parse_args()
    main(args)
