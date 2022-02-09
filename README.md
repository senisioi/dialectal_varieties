# Dialectal Varieties

## Directories:
    
- corpus/train_test_split: the train test splits for comparing classification similarities across languages
- analogies: csv files of misaligned words across corpora
- feature_selection: directory, we saved the accuracy, F1 score and cofusion matrices for each classification scenario, filenames: FEATURE_LANGUAGE_PAIRS.ftrs

## Corpus:

- Contains also the EU documents
- Each document has 66 sentences of ~ 2000 words
- The directory chnk contains each data point on a line
- The original indices before shuffling are stored in *.orig_idx
- Shuffled documents are found in *.shf

Each test set contains 40 documents from each class, and the training sets have a maximum of 197 documents (the minimum nr given Scotland, the smallest class)


Corpus size, mean sentence length and standard deviation, nr of types, tokens and type/token ratio:
```
    Scotland_en   15646  mean sent len:  (26.023712130896076, 13.597490300686907)   (15792, 407167, 3.8785068534532514)
    England_en   60179  mean sent len:  (26.40693597434321, 13.8278072902495)   (28003, 1589143, 1.7621447534929204)
    Ireland_en    31443  mean sent len:  (26.094170403587444, 13.063950360254797)   (20089, 820479, 2.448447796957631)
    EU_en    29693  mean sent len:  (26.356447647593708, 14.036353105547606)   (18401, 782602, 2.3512590052159337)
    Wales_en    2466  mean sent len:  (25.763584752635847, 12.307185570675403)   (5671, 63533, 8.926069916421387)
    Unknown_en    6607  mean sent len:  (25.8471318298774, 13.193737496207815)   (9897, 170772, 5.795446560326049)
    Scotland_fr    15646  mean sent len:  (29.997699092419786, 16.145082183578896)   (20056, 469344, 4.273198336401445)
    England_fr    60179  mean sent len:  (30.015121554030475, 16.077963478296283)   (35351, 1806280, 1.9571162831897602)
    Ireland_fr    31443  mean sent len:  (29.723849505454314, 15.263570960169266)   (25516, 934607, 2.7301314884224066)
    EU_fr    29734  mean sent len:  (29.770027577857, 16.442047863195405)   (23544, 885182, 2.659791997577899)
    Wales_fr    2466  mean sent len:  (29.33049472830495, 14.713038377418508)   (7268, 72329, 10.04852825284464)
    Unknown_fr    6607  mean sent len:  (29.193128500075677, 15.26003637482646)   (12718, 192879, 6.593771224446415)
```

Maximum nr of sents / doc  66
Scotland chunks:  237
England chunks:  911
Ireland chunks:  476
EU chunks:  449


