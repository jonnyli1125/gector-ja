# gector-ja

Grammatical error correction model described in the paper ["GECToR -- Grammatical Error Correction: Tag, Not Rewrite" (Omelianchuk et al. 2020)](https://arxiv.org/abs/2005.12592), implemented for Japanese. This project's code is based on the official implementation (https://github.com/grammarly/gector).

The [pretrained Japanese BERT model](https://huggingface.co/cl-tohoku/bert-base-japanese-v2) used in this project was provided by Tohoku University NLP Lab.

## Datasets

- [Japanese Wikipedia dump](https://dumps.wikimedia.org/), extracted with [WikiExtractor](https://github.com/attardi/wikiextractor), synthetic errors generated using preprocessing scripts
  - 19,841,767 training sentences
- [NAIST Lang8 Learner Corpora](https://sites.google.com/site/naistlang8corpora/)
  - 6,066,306 training sentences (generated from 3,084,0376 original sentences)

### Synthetically Generated Error Corpus

The Wikipedia corpus was used to synthetically generate errorful sentences, with a method similar to [Awasthi et al. 2019](https://github.com/awasthiabhijeet/PIE/tree/master/errorify), but with adjustments for Japanese. The details of the implementation can be found in the [preprocessing scripts](https://github.com/jonnyli1125/gector-ja/blob/main/utils/) in this repository.

Example error-generated sentence:
```
西口側には宿泊施設や地元の日本酒や海、山の幸を揃えた飲食店、呑み屋など多くある。        # Correct
西口側までは宿泊から施設や地元の日本酒や、山の幸を揃えた飲食は店、呑み屋など多くあろう。 # Errorful
```

### Edit Tagging

Using the preprocessed Wikipedia corpus and Lang8 corpus, the errorful sentences were tokenized using the WordPiece tokenizer from the [pretrained BERT model](https://huggingface.co/cl-tohoku/bert-base-japanese-v2). Each token was then mapped to a minimal sequence of token transformations, such that when the transformations are applied to the errorful sentence, it will lead to the target sentence. The GECToR paper explains this preprocessing step in more detail (section 3), and the code specifics can be found in the [official implementation](https://github.com/grammarly/gector/blob/master/utils/preprocess_data.py).

Example edit-tagged sentence (using the same pair of sentences above):
```
[CLS] 西口  側    まで         は    宿泊  から     施設  や    地元  の     日本  酒    や         、    山    の     幸    を    揃え  た     飲食  は      店    、    呑     ##み  ##屋  など   多く  あろう             。    [SEP]
$KEEP $KEEP $KEEP $REPLACE_に $KEEP $KEEP $DELETE $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $APPEND_海 $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $DELETE $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $KEEP $TRANSFORM_VBV_VB $KEEP $KEEP
```

Furthermore, on top of the basic 4 token transformations (`$KEEP`, `$DELETE`, `$APPEND`, `$REPLACE`), there are a set of special transformations called "g-transformations" (i.e. `$TRANSFORM_VBV_VB` in the example above). G-transformations are mainly used for common replacements, such as switching verb conjugations, as described in the GECToR paper (section 3). The g-transformations in this model were redefined to accommodate for Japanese verbs and i-adjectives, which both inflect for tense.

## Model Architecture

The model consists of a [pretrained BERT encoder layer](https://huggingface.co/cl-tohoku/bert-base-japanese-v2) and two linear classification heads, one for `labels` and one for `detect`. `labels` predicts a specific edit transformation (`$KEEP`, `$DELETE`, `$APPEND_x`, etc), and `detect` predicts whether the token is `CORRECT` or `INCORRECT`. The results from the two are used to make a prediction. The predicted transformations are then applied to the errorful input sentence to obtain a corrected sentence.

Furthermore, in some cases, one pass of predicted transformations is not sufficient to transform the errorful sentence to the target sentence. Therefore, we repeat the process again on the result of the previous pass of transformations, until the model predicts that the sentence no longer contains incorrect tokens.

For more details about the model architecture and __iterative sequence tagging approach__, refer to section 4 and 5 of the GECToR paper or the [official implementation](https://github.com/grammarly/gector/blob/master/gector/seq2labels_model.py).

## Training

The model was trained in Colab with TPUs on each corpus with the following hyperparameters (default is used if unspecified):

```
batch_size: 64
learning_rate: 1e-5
bert_trainable: true
```

Synthetic error corpus (Wikipedia dump):
```
length: 19841767
epochs: 3
```

Lang8 corpus:
```
length: 6066306
epochs: 10
```

## Demo App

Trained weights can be downloaded [here](https://drive.google.com/file/d/1nhWzDZnZKxLvqwYMLlwRNOkMK2aXv4-5/view?usp=sharing).

Extract `model.zip` to the `data/` directory. You should have the following folder structure:

```
gector-ja/
  data/
    model/
      checkpoint
      model_checkpoint.data-00000-of-00001
      model_checkpoint.index
    ...
  main.py
  ...
```

After downloading and extracting the weights, the demo app can be run with the command `python main.py`.

You may need to `pip install flask` if Flask is not already installed.

## Evaluation

The model can be evaluated with `evaluate.py` on a parallel sentences corpus. The evaluation corpus used was [TMU Evaluation Corpus for Japanese Learners (Koyama et al. 2020)](https://www.aclweb.org/anthology/2020.lrec-1.26/), and the metric is GLEU score.

Using the model trained with the parameters described above, it achieved a GLEU score of around 0.81, which appears to outperform the CNN-based method by Chollampatt and Ng, 2018 (state of the art on the CoNLL-2014 dataset prior to transformer-based models), that Koyama et al. 2020 chose to use in their paper.

#### CoNLL-2014 (GEC dataset for English)
| Method                    | F0.5  |
| ------------------------- | ----- |
| Chollampatt and Ng, 2018  | 56.52 |
| Omelianchuk et al., 2020  | 66.5  |

#### TMU Evaluation Corpus for Japanese Learners (GEC dataset for Japanese)
| Method                    | GLEU  |
| ------------------------- | ----- |
| Chollampatt and Ng, 2018  | 0.739 |
| __gector-ja (this project)__  | __0.81__  |

In the GECToR paper, F0.5 score was used, which can also be determined through use of [errant](https://github.com/chrisjbryant/errant) and [m2scorer](https://github.com/nusnlp/m2scorer). However, these tools were designed to be used for evaluation on the CoNLL-2014 dataset, and using them for this project would also require modifying the tools' source code to accommodate for Japanese. In this project GLEU score was used as in Koyama et al. 2020, which works "out of the box" from the NLTK library.
