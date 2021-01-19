import os

bert = "bert-base-cased"
bert_nl = "wietsedv/bert-base-dutch-cased"
roberta = "roberta-base"
roberta_nl = "pdelobelle/robbert-v2-dutch-base"
mbert = "bert-base-multilingual-cased"

data_folder = 'sick_nl/data'
stress_test_folder = os.path.join(data_folder, 'tasks/stress_tests')
sick_folder = os.path.join(data_folder, 'tasks/sick_nl')

sick_path = os.path.join(sick_folder, 'SICK.txt')
sicknl_path = os.path.join(sick_folder, 'SICK_NL.txt')

skipgram_fn = os.path.join(data_folder, 'GoogleNews-vectors-negative300_SICK.txt')
skipgram_fn_nl = os.path.join(data_folder, '320/wikipedia-320.txt')

sick_fn = os.path.join(data_folder, 'SICK.txt')
sick_fn_nl = os.path.join(data_folder, 'SICK_NL.txt')

models_folder = os.path.join(data_folder, 'models')
results_folder = os.path.join(data_folder, 'results')

prep_order_fn = os.path.join(data_folder, 'tasks/stress_tests/prep_phrase_order.txt')
present_tense_fn =
