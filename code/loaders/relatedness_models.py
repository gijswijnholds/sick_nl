from sick_nl.code.config import (skipgram_fn, skipgram_fn_nl, bert,
                                 bert_nl, roberta, roberta_nl)
from sick_nl.code.models.skipgram import AverageSkipgramModel
from sick_nl.code.models.bert_basic import GenericBERT


def load_models_en():
    print("Loading skipgram model...")
    skipgram_model = AverageSkipgramModel(skipgram_fn)
    print("Loading BERT models...")
    bert_cls = GenericBERT(bert, 'bert', setting='cls')
    bert_avg = GenericBERT(bert, 'bert', setting='average')
    print("Loading ROBERTA models...")
    roberta_cls = GenericBERT(roberta, 'roberta', setting='cls')
    roberta_avg = GenericBERT(roberta, 'roberta', setting='average')
    models = [('skipgram', skipgram_model), ('bert-cls', bert_cls),
              ('bert-avg', bert_avg), ('roberta-cls', roberta_cls),
              ('roberta-avg', roberta_avg)]
    return models


def load_models_nl():
    print("Loading skipgram model...")
    skipgram_model = AverageSkipgramModel(skipgram_fn_nl)
    print("Loading BERTJE models...")
    bertje_cls = GenericBERT(bert_nl, 'bert', setting='cls')
    bertje_avg = GenericBERT(bert_nl, 'bert', setting='average')
    print("Loading ROBBERT models...")
    robbert_cls = GenericBERT(roberta_nl, 'roberta', setting='cls')
    robbert_avg = GenericBERT(roberta_nl, 'roberta', setting='average')
    models = [('skipgram', skipgram_model), ('bertje-cls', bertje_cls),
              ('bertje-avg', bertje_avg), ('robbert-cls', robbert_cls),
              ('robbert-avg', robbert_avg)]
    return models
