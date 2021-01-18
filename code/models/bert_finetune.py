"""BERT fine-tuning model, where we train a classifier on top of the [CLS]
vector for a pair of sentences to do NLI predictions."""
import math
import shutil
import torch
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import List, Dict, Any
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'trues': labels,
        'predictions': preds
    }


def my_data_collator(features: List[Any]) -> Dict[str, torch.Tensor]:
    """
    A nearly identical version of the default_data_collator from the
    HuggingFace transformers library.
    """
    first = features[0]
    batch = {}
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)
    return batch


def load_data_as_features(data, tokenizer):
    classes = {"CONTRADICTION": 0., "NEUTRAL": 1., "ENTAILMENT": 2.}
    all_data = []
    for (s1, s2, el, rl) in data:
        di = tokenizer(s1.lower(), s2.lower(), max_length=121, padding='max_length')
        # having a max_length of 121 was fine, but with cross-lingual stuff the tokenisation will get messed up so need to increase the max length to 128
        di['label'] = torch.tensor(classes[el], dtype=torch.long)
        all_data.append(di)
    return all_data


class SICK_BERT_DATASET(Dataset):
    def __init__(self, data, tokenizer):
        self.data = load_data_as_features(data, tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class BERTFineTuner(object):
    def __init__(self, name, tokenizer, model, train_dataset, test_dataset,
                 num_epochs=3, freeze=False):
        self.model_name = name.split('/')[-1]
        self.tokenizer = tokenizer
        self.model = model
        self.freeze = freeze
        if self.freeze:
            for p in self.model.base_model.parameters():
                p.requires_grad = False
            self.model_name = self.model_name + '_frozen'
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_epochs = num_epochs
        self.training_args = self.setup_train_args()
        self.trainer = self.setup_trainer()

    def setup_train_args(self):
        batch_size = 16
        step_eval = math.ceil(len(self.train_dataset)/2/batch_size)
        outdir_name = f'./results2_{self.model_name}'
        logdir_name = f'./logs2_{self.model_name}'
        if os.path.exists(outdir_name):
            shutil.rmtree(outdir_name)
        if os.path.exists(logdir_name):
            shutil.rmtree(logdir_name)
        return TrainingArguments(
            output_dir=outdir_name,
            save_total_limit=1,
            # save_steps=0,
            num_train_epochs=self.num_epochs,
            # evaluate_during_training=True,
            eval_steps=step_eval,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            warmup_steps=250,
            weight_decay=0.01,
            logging_dir=logdir_name,
        )

    def setup_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=my_data_collator,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=compute_metrics
        )

    def train(self):
        return self.trainer.train()

    def evaluate(self):
        return self.trainer.evaluate()
