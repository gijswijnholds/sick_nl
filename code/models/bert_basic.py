"""BERT vector aggregate model that takes CLS vector or average vector."""
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel


class GenericBERT():
    def __init__(self, name, sort='bert', setting='average'):
        self.setting = setting
        assert sort in ['bert', 'roberta']
        if sort == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(name)
            self.model = BertModel.from_pretrained(name)
        elif sort == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(name)
            self.model = RobertaModel.from_pretrained(name)

    def model_sentence(self, sentence: str) -> np.ndarray:
        tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence.lower()) + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tens = torch.LongTensor(input_ids).unsqueeze(0)
        all_vecs = self.model(tens)[0].squeeze()
        class_vec, sent_vecs = all_vecs[0], all_vecs[1:-1]
        if self.setting == 'cls':
            return class_vec.detach().numpy()
        elif self.setting == 'average':
            return np.average(sent_vecs.detach().numpy(), axis=0)
