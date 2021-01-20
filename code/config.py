import os

bert = "bert-base-cased"
bert_nl = "wietsedv/bert-base-dutch-cased"
roberta = "roberta-base"
roberta_nl = "pdelobelle/robbert-v2-dutch-base"
mbert = "bert-base-multilingual-cased"

data_folder = 'sick_nl/data'
stress_test_folder = os.path.join(data_folder, 'tasks/stress_tests')
sick_folder = os.path.join(data_folder, 'tasks/sick_nl')
vectors_folder = os.path.join(data_folder, 'vectors')

sick_fn_en = os.path.join(sick_folder, 'SICK.txt')
sick_fn_nl = os.path.join(sick_folder, 'SICK_NL.txt')

skipgram_fn = os.path.join(vectors_folder, 'GoogleNews-vectors-negative300_SICK.txt')
skipgram_fn_nl = os.path.join(vectors_folder, '320/wikipedia-320.txt')

model_data_folder = 'sick_nl/model_data'
models_folder = os.path.join(model_data_folder, 'models')
models_agg_folder = os.path.join(model_data_folder, 'models_agg')
results_folder = os.path.join(model_data_folder, 'results')

prep_order_fn = os.path.join(stress_test_folder, 'prep_phrase_order.txt')
present_tense_fn = os.path.join(stress_test_folder, 'present_cont_present_simple.txt')

best_bert_nl_model = os.path.join(models_agg_folder, 'model_SICK_NL_bert-base-dutch-cased_epoch16.pt')
best_roberta_nl_model = os.path.join(models_agg_folder, 'model_SICK_NL_robbert-v2-dutch-base_epoch19.pt')
best_mbert_nl_model = os.path.join(models_agg_folder, 'model_SICK_NL_bert-base-multilingual-cased_epoch16.pt')
