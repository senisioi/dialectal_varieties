import os
from text_classifications import *

rows = []

# English experiments

inp_dir = '../corpus/train_test_split/en'
feature_paths = [None, '../features/function_words_en', '../features/pronouns_en']
for feature_path in feature_paths:
    row = {}
    row['setup'] = 'English Originals'

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



inp_dir = "../corpus/train_test_split/en_pos"
if os.path.exists(inp_dir):
    feature_paths = [None, '../features/selected_pos_ngrams_en', '../features/selected_pos_ngrams_fr']
    for feature_path in feature_paths:
        row = {}
        row['setup'] = 'English Originals'
        if feature_path is None:
            row['feature'] = 'PoS n-grams'
        else:   
            row['feature'] = os.path.basename(feature_path)
        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS,  ['England', 'Ireland'])
        row['En vs. Ir'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, ['England', 'Scotland'])
        row['En vs. Sc'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, ['Ireland', 'Scotland'])
        row['Ir vs. Sc'] = avg_f1

        avg_f1, _, _ = run_dialect_experiment_multi_sample(inp_dir, feature_path, MAX_NO_POS_NGRAMS, [])
        row['3-way'] = avg_f1
        rows.append(row)


inp_dir = '../corpus/train_test_split/en_noent'
feature_path = None
if os.path.exists(inp_dir):
    row = {}
    row['setup'] = 'English no Entities'
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
df.to_csv('english.csv', index=False)