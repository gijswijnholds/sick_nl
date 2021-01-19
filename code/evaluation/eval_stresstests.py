import torch
from transformers import BertTokenizer, RobertaTokenizer
from sick_nl.code.models.bert_finetune import SICK_BERT_DATASET, BERTFineTuner
from sick_nl.code.config import prep_order_fn, present_tense_fn

"""TODO: evaluate on the stress test itself, and evaluate on SICK-test pairs with
    replacement of stress test rewrites."""


def load_stress_test(test_fn: str):
    with open(test_fn) as inf:
        data = [ln.split('\t') for ln in inf.readlines()[1:]]
    return data


def load_prep_order():
    return load_stress_test(prep_order_fn)


def load_present_tense():
    return load_stress_test(present_tense_fn)


def evaluate_trained_model(old_pairs, new_pairs, fn, name, setting='bert'):
    print("Loading model...")
    if setting == 'bert':
        tokenizer = BertTokenizer.from_pretrained(name)
    elif setting == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(name)
    model = torch.load(fn)
    print("Loading datasets...")
    old_dataset = SICK_BERT_DATASET(old_pairs, tokenizer)
    new_dataset = SICK_BERT_DATASET(new_pairs, tokenizer)
    print("Loading finetuning model...")
    old_tuner = BERTFineTuner(name, tokenizer, model, old_dataset, old_dataset,
                              num_epochs=1, freeze=False)
    new_tuner = BERTFineTuner(name, tokenizer, model, new_dataset, new_dataset,
                              num_epochs=1, freeze=False)
    old_results = old_tuner.evaluate()
    new_results = new_tuner.evaluate()
    return old_results, new_results


def evaluate_on_stress_tests():
    pass
