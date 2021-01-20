from sick_nl.code.config import (bert_nl, best_bert_nl_model,
                                 roberta_nl, best_roberta_nl_model,
                                 mbert, best_mbert_nl_model)
from sick_nl.code.loaders.stress_tests import (load_prep_order, load_present_tense,
                                               load_switched_sick, load_stress_test_bidirectional)
from sick_nl.code.loaders.nli_models import load_bert_nli_model_stress_test


def evaluate_switched_sick(data, model_fn, name, setting):
    data_before, data_after = load_switched_sick(data)
    results_before = load_bert_nli_model_stress_test(data_before, model_fn, name, setting).evaluate()
    results_after = load_bert_nli_model_stress_test(data_after, model_fn, name, setting).evaluate()
    return results_before, results_after


def evaluate_stress_test(data, model_fn, name, setting):
    left_to_right_data, right_to_left_data = load_stress_test_bidirectional(data)
    results_left_to_right = load_bert_nli_model_stress_test(left_to_right_data, model_fn, name, setting).evaluate()
    results_right_to_left = load_bert_nli_model_stress_test(right_to_left_data, model_fn, name, setting).evaluate()
    return results_left_to_right, results_right_to_left


def evaluate_switched_sicks():
    prep_order_data, pres_tense_data = load_prep_order(), load_present_tense()
    results = {'bert_nl': {}, 'roberta_nl': {}, 'mbert': {}}
    results['bert_nl']['prep_order'] = evaluate_switched_sick(prep_order_data, best_bert_nl_model, bert_nl, setting='bert')
    results['roberta_nl']['prep_order'] = evaluate_switched_sick(prep_order_data, best_roberta_nl_model, roberta_nl, setting='roberta')
    results['mbert']['prep_order'] = evaluate_switched_sick(prep_order_data, best_mbert_nl_model, mbert, setting='bert')
    results['bert_nl']['pres_tense'] = evaluate_switched_sick(pres_tense_data, best_bert_nl_model, bert_nl, setting='bert')
    results['roberta_nl']['pres_tense'] = evaluate_switched_sick(pres_tense_data, best_roberta_nl_model, roberta_nl, setting='roberta')
    results['mbert']['pres_tense'] = evaluate_switched_sick(pres_tense_data, best_mbert_nl_model, mbert, setting='bert')
    return results


def evaluate_stress_tests():
    prep_order_data, pres_tense_data = load_prep_order(), load_present_tense()
    results = {'bert_nl': {}, 'roberta_nl': {}, 'mbert': {}}
    results['bert_nl']['prep_order'] = evaluate_stress_test(prep_order_data, best_bert_nl_model, bert_nl, setting='bert')
    results['roberta_nl']['prep_order'] = evaluate_stress_test(prep_order_data, best_roberta_nl_model, roberta_nl, setting='roberta')
    results['mbert']['prep_order'] = evaluate_stress_test(prep_order_data, best_mbert_nl_model, mbert, setting='bert')
    results['bert_nl']['pres_tense'] = evaluate_stress_test(pres_tense_data, best_bert_nl_model, bert_nl, setting='bert')
    results['roberta_nl']['pres_tense'] = evaluate_stress_test(pres_tense_data, best_roberta_nl_model, roberta_nl, setting='roberta')
    results['mbert']['pres_tense'] = evaluate_stress_test(pres_tense_data, best_mbert_nl_model, mbert, setting='bert')
    return results
