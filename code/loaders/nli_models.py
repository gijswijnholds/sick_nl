from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sick_nl.code.models.bert_finetune import SICK_BERT_DATASET, BERTFineTuner


def load_bert_nli_model(sick_dataset, name, setting):
    print("Loading BERT model...")
    if setting == 'bert':
        tokenizer = BertTokenizer.from_pretrained(name)
        model = BertForSequenceClassification.from_pretrained(name, num_labels=3)
    elif setting == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(name)
        model = RobertaForSequenceClassification.from_pretrained(name, num_labels=3)
    print("Preparing datasets...")
    train_dataset = SICK_BERT_DATASET(sick_dataset.train_data, tokenizer)
    eval_dataset = SICK_BERT_DATASET(sick_dataset.test_data, tokenizer)
    print("Loading finetuning model...")
    tuner = BERTFineTuner(name, tokenizer, model, train_dataset, eval_dataset,
                          num_epochs=1, freeze=False)
    return tuner


def load_bert_nli_model_stress_test(sick_dataset, name, setting):
    pass
