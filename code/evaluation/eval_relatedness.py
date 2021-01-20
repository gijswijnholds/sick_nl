"""Evaluate a model on the SICK-NL dataset."""
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from sick_nl.code.loaders.sick import load_sick_nl, load_sick_en
from sick_nl.code.loaders.relatedness_models import load_models_nl, load_models_en


class Evaluator(object):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def evaluate(self, setting='test'):
        if setting == 'test':
            pred_data = self.dataset.test_data
        elif setting == 'train':
            pred_data = self.dataset.train_data
        elif setting == 'dev':
            pred_data = self.dataset.dev_data

        preds = [cosine_similarity([self.model.model_sentence(s1),
                                    self.model.model_sentence(s2)])[0][1]
                 for (s1, s2, _, _) in tqdm(pred_data)]
        trues = [rl for (_, _, _, rl) in pred_data]
        spearman_b, pearson_b = spearmanr(preds, trues), pearsonr(preds, trues)
        return round(100*spearman_b[0], 2), round(100*pearson_b[0], 2)


def evaluate_baseline_models(models, dataset):
    print("Evaluating baseline models...")
    evaluation_results = {}
    for (m_name, m) in models:
        print(f"Evaluating {m_name}...")
        evaluation_results[m_name] = Evaluator(dataset, m).evaluate()
    return evaluation_results


def evaluate_nl_models():
    print("Loading models...")
    models = load_models_nl()
    print("Loading NL SICK...")
    nl_sick = load_sick_nl()
    eval_results = evaluate_baseline_models(models, nl_sick)
    return eval_results


def evaluate_en_models():
    print("Loading models...")
    models = load_models_en()
    print("Loading SICK...")
    en_sick = load_sick_en()
    eval_results = evaluate_baseline_models(models, en_sick)
    return eval_results
