# Dialectal Varieties

## Corpus:

- Contains also the EU documents
- Each document has 66 sentences of `~2000` words
- The `*.chnk` files contain each data point on a line with sentences split by '#%'
- The original indices before shuffling are stored in `*.orig_idx`
- Shuffled documents are found in `*.shf`

Each test set contains 40 documents from each class, and the training sets have a maximum of 197 documents (the minimum nr given Scotland, the smallest class)


## Running the code

### 1. Install requirements

Install requirements. Spacy is only needed for to obtain PoS tags.

```bash
pip3 install requirements.txt
```

Download spacy models for English and French.
```bash
# you may choose to download other models if you wish so
python3 -m spacy download en_core_web_lg
python3 -m spacy download fr_core_news_lg
```

##### Optional - split raw files into chunks
This is not needed, as the splits are already released in this repository. If one wishes to split the files into chunks and to do the train-test split, run:
```bash
python3 split_extract.py
```

##### Optional - machine translate English documents to French
We used an En-Fr readily-available MT model implemented in [fairseq-py](https://github.com/pytorch/fairseq). [The fairseq model](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) is available for download and may be loaded into fairseq by following [the tutorial from the official repository](https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md). The MT model is only important to show that dialect-markers from the source language are preserved in the MT output.




### 2. Generate pos-tagged directories

This will generate PoS-tagged directories from English, French, and MT French train-test splits. Check the source code, if you wish to run it on a custom directory.
```bash
cd src && python3 make_pos_dirs.py
``` 

### 3. Generate some tables from text classification experiments

This will run all the configurations in the `*_classifications.py` scripts. Currently it does classifications using function words, pronouns, PoS n-grams, word n-grams.
```bash
cd src && \
python3 en_classifications.py && \
python3 fr_classifications.py
``` 
The classification scores are obtained by doing multiple experiments on downsampled data. Each class is split into batches of equal size and for each split, we train-test a classifier.


## Directories:

- corpus/* : directories of texts produced by native, non-native MEP's
- corpus/train_test_split: the train test splits for comparing classification similarities across languages
- analogies: csv files of misaligned words across corpora
- feature_selection: directory, we saved the accuracy, F1 score and cofusion matrices for different classification scenarios, filenames: FEATURE_LANGUAGE_PAIRS.ftrs
- features: lists of word and PoS features 
- src: source file directory


Maximum nr of sents / doc  66
Scotland chunks:  237
England chunks:  911
Ireland chunks:  476
EU chunks:  449


## Results

### Original English Data

|setup|feature                      |En vs. Ir|En vs. Sc                                    |Ir vs. Sc|3-way|
|-----|-----------------------------|---------|---------------------------------------------|---------|-----|
|English Originals|PoS n-grams                  |0.96     |0.92                                         |0.92     |0.89 |
|English Originals|selected_pos_ngrams_en       |0.9      |0.87                                         |0.87     |0.79 |
|English Originals|Word n-grams                 |0.99     |1                                            |0.99     |0.99 |
|English Originals|function_words_en            |0.9      |0.91                                         |0.87     |0.79 |
|English Originals|pronouns_en                  |0.68     |0.75                                         |0.69     |0.57 |



### French Human and Machine Translated Data

|setup|feature                      |En vs. Ir|En vs. Sc                                    |Ir vs. Sc|3-way|
|-----|-----------------------------|---------|---------------------------------------------|---------|-----|
|French Translations|PoS n-grams                  |0.89     |0.85                                         |0.82     |0.77 |
|French Translations|selected_pos_ngrams_fr       |0.78     |0.74                                         |0.75     |0.66 |
|French Translations|Word n-grams                 |1        |0.95                                         |0.99     |0.94 |
|French Translations|function_words_fr            |0.81     |0.87                                         |0.82     |0.73 |
|French Translations|pronouns_fr                  |0.81     |0.83                                         |0.72     |0.68 |
|French MT|PoS n-grams                  |0.96     |0.9                                          |0.86     |0.83 |
|French MT|selected_pos_ngrams_fr       |0.79     |0.74                                         |0.79     |0.67 |
|French MT|Word n-grams                 |1        |0.95                                         |0.99     |0.94 |
|French MT|function_words_fr            |0.87     |0.83                                         |0.8      |0.72 |
|French MT|pronouns_fr                  |0.86     |0.85                                         |0.72     |0.68 |
