'''
Utility script to chunk, shuffle and split the data into train/test.
At the end, it prints some basic statistics on the screen.
'''

import sys
import os
import numpy as np


def read_entire_file(fis):
    with open(fis, "r", "utf-8", encoding='utf-8') as fin:
        return np.array(fin.readlines())


def dump_to_file(fis, sents):
    with open(fis, "w", "utf-8", encoding='utf-8') as fin:
        for sent in sents:
            fin.write(sent)


def chunk_data(data):
    chunks = []
    end_of_data = int(len(data) / SENTS_PER_DOC) * SENTS_PER_DOC
    for idx in range(0, end_of_data, SENTS_PER_DOC):
        chunks.append(data[idx : idx + SENTS_PER_DOC])
    return chunks


def chunks_to_array(chunks):
    return np.hstack([chunk for chunk in chunks])


def dump_chunks_to_file(fis, chunks_of_sents, separator = "#%"):
    with codecs.open(fis, "w", "utf-8") as fin:
        for chunk in chunks_of_sents:
            for sent in chunk:
                # adding sentence separator for recovery and extra feature
                fin.write(sent.strip() + " " + separator + " ")
            fin.write("\n")


def dump_train_test_chunk(path, label, chunks, test_size, min_train_size):
    dump_chunks_to_file(os.path.join(path, label + ".chnk.test"), chunks[:test_size])
    for idx in range(test_size, len(chunks), min_train_size):
        dump_chunks_to_file(os.path.join(path, label + ".chnk." + str(idx - test_size) + ".train"), chunks[idx : idx + min_train_size])


def dump_train_test_sent(path, label, chunks, test_size, min_train_size):
    dump_to_file(os.path.join(path, label + ".snt.test"), chunks_to_array(chunks[:test_size]))
    for idx in range(test_size, len(chunks), min_train_size):
        dump_to_file(os.path.join(path, label + ".snt." + str(idx - test_size) + ".train"), chunks_to_array(chunks[idx : idx + min_train_size]))


def mean_sent_length(sents):
    sts = [len(sent.split()) for sent in sents]
    return (np.mean(sts), np.std(sts))


def type_token(sents):
    split = [sent.split() for sent in sents]
    wds = [wd.lower() for sent in split for wd in sent]
    return (len(set(wds)), len(wds), 100*len(set(wds))/len(wds))


def shuffle_dump_indices(data, where_to_save):
    arr = np.arange(len(data))
    np.random.shuffle(arr)
    with open(where_to_save, 'w') as fout:
        for el in arr:
            fout.write(str(el) + '\n')
    return arr  


def print_stats(label, data):
    print (label, " ", len(data), " mean sent len: ", mean_sent_length(data), " ", type_token(data))


EN_Natives = "../corpus/natives_en"
FR_Natives = "../corpus/natives_fr"
EN_EU = "../corpus/nonnatives_en"
FR_EU = "../corpus/nonnatives_fr"


scots_en = read_entire_file(os.path.join(EN_Natives, "Scotland"))
engl_en = read_entire_file(os.path.join(EN_Natives, "England"))
irl_en = read_entire_file(os.path.join(EN_Natives, "Ireland"))
eu_en = read_entire_file(os.path.join(EN_EU, "EU"))


print_stats("Scotland_en", scots_en)
print_stats("England_en", engl_en)
print_stats("Ireland_en ", irl_en)
print_stats("EU_en ", eu_en)
print_stats("Wales_en ", read_entire_file(os.path.join(EN_Natives, "Wales")))
print_stats("Unknown_en ", read_entire_file(os.path.join(EN_Natives, "Unknown")))


scots_fr = read_entire_file(os.path.join(FR_Natives, "Scotland"))
engl_fr = read_entire_file(os.path.join(FR_Natives, "England"))
irl_fr = read_entire_file(os.path.join(FR_Natives, "Ireland"))
eu_fr = read_entire_file(os.path.join(FR_EU, "EU"))

print_stats("Scotland_fr ", scots_fr)
print_stats("England_fr ", engl_fr)
print_stats("Ireland_fr ", irl_fr)
print_stats("EU_fr ", eu_fr)
print_stats("Wales_fr ", read_entire_file(os.path.join(FR_Natives, "Wales")))
print_stats("Unknown_fr ", read_entire_file(os.path.join(FR_Natives, "Unknown")))


print ("shuffle dump idx...")
scots_idx = shuffle_dump_indices(scots_en, os.path.join(EN_Natives, "Scotland.orig_idx"))
scots_en = scots_en[scots_idx]
scots_fr = scots_fr[scots_idx]

engl_idx = shuffle_dump_indices(engl_en, os.path.join(EN_Natives, "England.orig_idx"))
engl_en = engl_en[engl_idx]
engl_fr = engl_fr[engl_idx]

irl_idx = shuffle_dump_indices(irl_en, os.path.join(EN_Natives, "Ireland.orig_idx"))
irl_en = irl_en[irl_idx]
irl_fr = irl_fr[irl_idx]

eu_idx = shuffle_dump_indices(eu_en, os.path.join(EN_EU, "EU.orig_idx"))
eu_en = irl_en[eu_idx]
eu_fr = irl_fr[eu_idx]

print ("dumping back to files...")
dump_to_file(os.path.join(EN_Natives, "Scotland.shf"), scots_en)
dump_to_file(os.path.join(EN_Natives, "England.shf"), engl_en)
dump_to_file(os.path.join(EN_Natives, "Ireland.shf"), irl_en)
dump_to_file(os.path.join(EN_EU, "EU.shf"), eu_en)
dump_to_file(os.path.join(FR_Natives, "Scotland.shf"), scots_fr)
dump_to_file(os.path.join(FR_Natives, "England.shf"), engl_fr)
dump_to_file(os.path.join(FR_Natives, "Ireland.shf"), irl_fr)
dump_to_file(os.path.join(FR_EU, "EU.shf"), eu_fr)


TEST_SIZE = 40
MAX_WDS = 2000
(m, _) = mean_sent_length(scots_fr)
SENTS_PER_DOC = int(MAX_WDS / m)
print ("Maximum nr of sents / doc ", SENTS_PER_DOC)


scots_en_chnk = chunk_data(scots_en)
scots_fr_chnk = chunk_data(scots_fr)
print ("Scotland chunks: ", len(scots_en_chnk))

engl_en_chnk = chunk_data(engl_en)
engl_fr_chnk = chunk_data(engl_fr)
print ("England chunks: ", len(engl_en_chnk))

irl_en_chnk = chunk_data(irl_en)
irl_fr_chnk = chunk_data(irl_fr)
print ("Ireland chunks: ", len(irl_en_chnk))

eu_en_chnk = chunk_data(eu_en)
eu_fr_chnk = chunk_data(eu_fr)
print ("EU chunks: ", len(eu_en_chnk))


MIN_TRAIN_SIZE = len(scots_en_chnk[TEST_SIZE:])


print ("dumping test files at chunk level...")
dump_train_test_chunk(EN_Natives, "Scotland", scots_en_chnk, TEST_SIZE, MIN_TRAIN_SIZE)
dump_train_test_chunk(FR_Natives, "Scotland", scots_fr_chnk, TEST_SIZE, MIN_TRAIN_SIZE)

dump_train_test_chunk(EN_Natives, "England", engl_en_chnk, TEST_SIZE, MIN_TRAIN_SIZE)
dump_train_test_chunk(FR_Natives, "England", engl_fr_chnk, TEST_SIZE, MIN_TRAIN_SIZE)

dump_train_test_chunk(EN_Natives, "Ireland", irl_en_chnk, TEST_SIZE, MIN_TRAIN_SIZE)
dump_train_test_chunk(FR_Natives, "Ireland", irl_fr_chnk, TEST_SIZE, MIN_TRAIN_SIZE)

dump_train_test_chunk(EN_EU, "EU", eu_en_chnk, TEST_SIZE, MIN_TRAIN_SIZE)
dump_train_test_chunk(FR_EU, "EU", eu_fr_chnk, TEST_SIZE, MIN_TRAIN_SIZE)


print ("dumping test files at sentence level...")
dump_train_test_sent(EN_Natives, "Scotland", scots_en_chnk, TEST_SIZE, MIN_TRAIN_SIZE)
dump_train_test_sent(FR_Natives, "Scotland", scots_fr_chnk, TEST_SIZE, MIN_TRAIN_SIZE)

dump_train_test_sent(EN_Natives, "England", engl_en_chnk, TEST_SIZE, MIN_TRAIN_SIZE)
dump_train_test_sent(FR_Natives, "England", engl_fr_chnk, TEST_SIZE, MIN_TRAIN_SIZE)

dump_train_test_sent(EN_Natives, "Ireland", irl_en_chnk, TEST_SIZE, MIN_TRAIN_SIZE)
dump_train_test_sent(FR_Natives, "Ireland", irl_fr_chnk, TEST_SIZE, MIN_TRAIN_SIZE)

dump_train_test_sent(EN_EU, "EU", eu_en_chnk, TEST_SIZE, MIN_TRAIN_SIZE)
dump_train_test_sent(FR_EU, "EU", eu_fr_chnk, TEST_SIZE, MIN_TRAIN_SIZE)

