"""Encapsulation of the SICK(NL) datasets."""
from sick_nl.code.config import sick_fn_en, sick_fn_nl


class SICK(object):
    def __init__(self, sick_fn: str):
        self.sick_fn = sick_fn
        self.name = self.sick_fn.split('/')[-1].split('.')[0]
        self.data = self.load_data()
        self.train_data, self.dev_data, self.test_data = self.split_data()

    def load_data(self):
        with open(self.sick_fn, 'r') as in_file:
            lines = [ln.strip().split('\t') for ln in in_file.readlines()][1:]
        sentence_data = [tuple(ln[1:5]+ln[-1:]) for ln in lines]
        sentence_data = [(s1, s2, el, float(rl), split)
                         for (s1, s2, el, rl, split) in sentence_data]
        return sentence_data

    def split_data(self):
        train_data, dev_data, test_data = [], [], []
        for (s1, s2, el, rl, s) in self.data:
            if s == 'TRAIN':
                train_data.append((s1, s2, el, rl))
            if s == 'TRIAL':
                dev_data.append((s1, s2, el, rl))
            if s == 'TEST':
                test_data.append((s1, s2, el, rl))
        return train_data, dev_data, test_data


def load_sick_en():
    return SICK(sick_fn_en)


def load_sick_nl():
    return SICK(sick_fn_nl)
