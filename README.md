# gector-ja

Grammatical error correction model described in the paper ["GECToR -- Grammatical Error Correction: Tag, Not Rewrite" (Omelianchuk et al. 2020)](https://arxiv.org/abs/2005.12592), implemented for Japanese.

This project's code is based on the official implementation (https://github.com/grammarly/gector).

TL;DR summary:
- GECToR (iterative sequence tagging GEC model) implemented using [pretrained BERT model for Japanese](https://huggingface.co/cl-tohoku/bert-base-japanese-v2) (from Tohoku University NLP Lab), `transformers==4.5.0`, and `AllenNLP==2.4.0` libraries.
- Synthetic error dataset generated from Wikipedia dump for pre-training (with similar method to [Awasthi et al. 2019](https://github.com/awasthiabhijeet/PIE/tree/master/errorify)), fine tuning done on NAIST Lang8 Learner Corpora.

## Datasets

- Pre-training: [Japanese Wikipedia dump](https://dumps.wikimedia.org/), extracted with [WikiExtractor](https://github.com/attardi/wikiextractor), synthetic errors generated using preprocessing scripts
  - 24,499,568 edit-tagged sentences
- Fine tuning: [NAIST Lang8 Learner Corpora](https://sites.google.com/site/naistlang8corpora/)
  - 6,039,088 edit-tagged sentences

### Synthetically Generated Error Corpus

The Wikipedia corpus was used to synthetically generate errorful sentences, with a method similar to [Awasthi et al. 2019](https://github.com/awasthiabhijeet/PIE/tree/master/errorify), but for Japanese. The details of the implementation can be found in the [preprocessing scripts](https://github.com/jonnyli1125/gector-ja/blob/main/utils/) in this repository.

Example synthetically error-generated sentence:
```
自然言拒以外については、人工言語・形式言語・コンピュータ言語など統で記事をが参照。 # Errorful
自然言語以外については、人工言語・形式言語・コンピュータ言語などの各記事を参照。   # Correct
```

### Edit Tagging

Using the preprocessed Wikipedia corpus and Lang8 corpus, the errorful sentences were tokenized using the WordPiece tokenizer from the [pretrained BERT model](https://huggingface.co/cl-tohoku/bert-base-japanese-v2). Each token was then mapped to a minimal sequence of token transformations, such that when the transformations are iteratively applied to the errorful sentence, it will lead to the target sentence. The GECToR paper explains this preprocessing step in more detail (section 3).

Example edit-tagged sentence (using the same pair of sentences above):
```
[CLS] 自然  言            拒      以外   に    つい  て    は    、    人工   言語  ・    形式   言語  ・    コンピュータ 言語  など  統          で          記事   を    が      参照  。    [SEP]
$KEEP $KEEP $REPLACE_言語 $DELETE $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP      $KEEP $KEEP $REPLACE_の $REPLACE_各 $KEEP $KEEP $DELETE $KEEP $KEEP $KEEP
```

Furthermore, on top of the basic 4 token transformations (`$KEEP`, `$DELETE`, `$APPEND`, `$REPLACE`), there are a set of special transformations called "g-transformations", described in the GECToR paper (section 3). G-transformations are mainly used for common replacements, such as switching verb conjugation forms. The g-transformations in this project were also redefined to accommodate for Japanese.

Lastly, in some cases, one pass of transformations is not sufficient to transform the errorful sentence to the target sentence. Therefore, we repeat the tagging process again on the result of the previous pass of transformations, until there are no more transformations to be applied. This means that for each pair of errorful and correct sentences, there may be more than one edit-tagged sentence generated (one for each iteration of transformations).

This __iterative sequence tagging approach__ is described in more detail in section 5 of the GECToR paper.

## Model Architecture

## Training

## Usage
