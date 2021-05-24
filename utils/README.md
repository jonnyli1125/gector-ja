# gector-ja/utils
- `preprocess_wiki_freqs.py`: Outputs JSON files that contain frequency of verbs, adjs, kanji, etc. given a Wikipedia dump.
- `preprocess_reading_lookup.py`: Outputs JSON file that contains a reading to kanji dictionary, given a KANJIDIC file.
- `preprocess_transformations.py`: Outputs a text file that contains all verb/i-adjective g-transformations, given the vocab list from frequency files.
- `preprocess_wiki.py`: Outputs correct sentences, errorful sentences, and edit-tagged errorful sentences given a Wikipedia dump. The previous scripts' outputs are required to run this script.
