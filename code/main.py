from sick_nl.code.evaluation.eval_relatedness \
    import evaluate_en_models, evaluate_nl_models
from sick_nl.code.evaluation.eval_nli \
    import evaluate_en_nli_models, evaluate_nl_nli_models
from sick_nl.code.evaluation.eval_stresstests \
    import evaluate_on_stress_tests


def main():
    evaluate_en_models()
    evaluate_nl_models()
    evaluate_en_nli_models()
    evaluate_nl_nli_models()
    evaluate_on_stress_tests()
