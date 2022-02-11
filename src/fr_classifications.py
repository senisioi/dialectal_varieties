import os
from text_classifications import *

rows = []

# French experiments - Human Translations

inp_dir = '../corpus/train_test_split/fr'
feature_paths = [None, '../features/function_words_fr', '../features/pronouns_fr']
for feature_path in feature_paths:
    row = {}
    row['setup'] = 'French Translations'

    if feature_path is None:
        row['feature'] = 'Word n-grams'
    else:   
        row['feature'] = os.path.basename(feature_path)

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['England', 'Ireland'])
    row['En vs. Ir'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['England', 'Scotland'])
    row['En vs. Sc'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['Ireland', 'Scotland'])
    row['Ir vs. Sc'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, [])
    row['3-way'] = avg_f1
    rows.append(row)


inp_dir = "../corpus/train_test_split/fr_pos"
if os.path.exists(inp_dir):
    feature_paths = [None, '../features/selected_pos_ngrams_fr', '../features/selected_pos_ngrams_en']
    for feature_path in feature_paths:
        row = {}
        row['setup'] = 'French Translations'
        if feature_path is None:
            row['feature'] = 'PoS n-grams'
        else:   
            row['feature'] = os.path.basename(feature_path)
        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, ['England', 'Ireland'])
        row['En vs. Ir'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, ['England', 'Scotland'])
        row['En vs. Sc'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, ['Ireland', 'Scotland'])
        row['Ir vs. Sc'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, [])
        row['3-way'] = avg_f1
        rows.append(row)


inp_dir = '../corpus/train_test_split/fr_noent'
feature_path = None
if os.path.exists(inp_dir):
    row = {}
    row['setup'] = 'French no Entities'
    row['feature'] = 'Word n-grams'

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['England', 'Ireland'])
    row['En vs. Ir'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['England', 'Scotland'])
    row['En vs. Sc'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['Ireland', 'Scotland'])
    row['Ir vs. Sc'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, [])
    row['3-way'] = avg_f1
    rows.append(row)


################################

# French experiments - Machine Translations

inp_dir = '../corpus/train_test_split/mt_fr'
feature_paths = [None, '../features/function_words_fr', '../features/pronouns_fr']
for feature_path in feature_paths:
    row = {}
    row['setup'] = 'French MT'

    if feature_path is None:
        row['feature'] = 'Word n-grams'
    else:   
        row['feature'] = os.path.basename(feature_path)

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['England', 'Ireland'])
    row['En vs. Ir'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['England', 'Scotland'])
    row['En vs. Sc'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['Ireland', 'Scotland'])
    row['Ir vs. Sc'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, [])
    row['3-way'] = avg_f1
    rows.append(row)


inp_dir = "../corpus/train_test_split/mt_fr_pos"
if os.path.exists(inp_dir):
    feature_paths = [None, '../features/selected_pos_ngrams_fr', '../features/selected_pos_ngrams_en']
    for feature_path in feature_paths:
        row = {}
        row['setup'] = 'French MT'
        if feature_path is None:
            row['feature'] = 'PoS n-grams'
        else:   
            row['feature'] = os.path.basename(feature_path)
        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, ['England', 'Ireland'])
        row['En vs. Ir'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, ['England', 'Scotland'])
        row['En vs. Sc'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, ['Ireland', 'Scotland'])
        row['Ir vs. Sc'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, [])
        row['3-way'] = avg_f1
        rows.append(row)


inp_dir = '../corpus/train_test_split/mt_fr_noent'
feature_path = None
if os.path.exists(inp_dir):
    row = {}
    row['setup'] = 'French no Entities'
    row['feature'] = 'Word n-grams'

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['England', 'Ireland'])
    row['En vs. Ir'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['England', 'Scotland'])
    row['En vs. Sc'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, ['Ireland', 'Scotland'])
    row['Ir vs. Sc'] = avg_f1

    avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_WD_NGRAMS, [])
    row['3-way'] = avg_f1
    rows.append(row)

###############################



df = pd.DataFrame(rows)
print(df)
df.to_csv('french.csv', index=False)