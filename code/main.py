from sick_nl.code.evaluation.eval_relatedness \
    import evaluate_en_models, evaluate_nl_models
from sick_nl.code.evaluation.eval_nli \
    import evaluate_en_nli_models, evaluate_nl_nli_models
from sick_nl.code.evaluation.eval_stresstests \
    import evaluate_switched_sicks, evaluate_stress_tests


def main():
    sick_en_relatedness_results = evaluate_en_models()
    sick_nl_relatedness_results = evaluate_nl_models()
    sick_en_nli_results = evaluate_en_nli_models()
    sick_nl_nli_results = evaluate_nl_nli_models()
    stress_test_results_insick = evaluate_switched_sicks()
    stress_test_results = evaluate_stress_tests()
