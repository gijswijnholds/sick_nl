import torch
from transformers import BertTokenizer, RobertaTokenizer
from sick_nl.code.models.bert_finetune import SICK_BERT_DATASET, BERTFineTuner


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
