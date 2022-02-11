import os
import spacy
from tqdm import tqdm

SENTENCE_SEP = '#%'


def files_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]


def make_dir(outp):
    if not os.path.exists(outp):
        print(f"Creating output directory in {outp}")
        os.makedirs(outp)


def spcy_pipe_process(texts, nlp, batch_size=50):
    docs = []
    for idx in range(0, len(texts), batch_size):
        tx_batch = [text.strip() for text in texts[idx:idx+batch_size]]
        docs.extend(nlp.pipe(tx_batch))
    return docs


def make_pos_dir(dir, spacy_lang, batch_size=80):
    '''PoS tagg all texts in a dir and save the result separate files.
    '''
    out_dir = dir.strip("/") + '_pos'
    make_dir(out_dir)
    texts = []
    for file in files_in_folder(dir):
        with open(file, 'r', encoding='utf-8') as fin:
            text = fin.read()
        texts.append((os.path.basename(file), text))
    nlp = spacy.load(spacy_lang)
    for fisname, text in texts:
        print('PoS tagging ', fisname)
        chunks = text.strip().split('\n')
        pos_chunks = []
        for chunk in tqdm(chunks):
            sentences = chunk.strip().split(SENTENCE_SEP)
            spcy_snts = spcy_pipe_process(sentences, nlp, batch_size=batch_size)
            pos_content = (' ' + SENTENCE_SEP + ' SNTSEP ').join([" ".join([t.pos_ for t in sentence]) for sentence in spcy_snts])
            pos_chunks.append(pos_content.strip())
        with open(os.path.join(out_dir, fisname), 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(pos_chunks))


def make_noent_dir(dir, spacy_lang, batch_size=80):
    '''Utility to make directory with no entities
    '''
    out_dir = dir.strip("/") + '_noent'
    make_dir(out_dir)
    texts = []
    for file in files_in_folder(dir):
        with open(file, 'r', encoding='utf-8') as fin:
            text = fin.read()
        texts.append((os.path.basename(file), text))
    nlp = spacy.load(spacy_lang)
    for fisname, text in texts:
        print('Entity tagging ', fisname)
        chunks = text.strip().split('\n')
        clean_chunks = []
        for chunk in tqdm(chunks):
            sentences = chunk.strip().split(SENTENCE_SEP)
            spcy_snts = spcy_pipe_process(sentences, nlp, batch_size=batch_size)
            clean_content = (' ' + SENTENCE_SEP + ' SNTSEP ').join([" ".join([t.text if t.ent_type == 0 else t.ent_type_  for t in sentence]) for sentence in spcy_snts])
            clean_chunks.append(clean_content.strip())
        with open(os.path.join(out_dir, fisname), 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(clean_chunks))


EN_SPCY_MODEL = 'en_core_web_trf'
FR_SPCY_MODEL = 'fr_dep_news_trf'

make_pos_dir('../corpus/train_test_split/en', EN_SPCY_MODEL)
make_pos_dir('../corpus/train_test_split/fr', FR_SPCY_MODEL)
make_pos_dir('../corpus/train_test_split/mt_fr', FR_SPCY_MODEL)

make_noent_dir('../corpus/train_test_split/en', EN_SPCY_MODEL)
make_noent_dir('../corpus/train_test_split/fr', FR_SPCY_MODEL)
make_noent_dir('../corpus/train_test_split/mt_fr', FR_SPCY_MODEL)


