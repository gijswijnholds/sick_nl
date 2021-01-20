# SICK-NL

A translation of the SICK dataset, for evaluating relatedness and entailment models for Dutch. SICK-NL was obtained by semi-automatically translating SICK (Marelli et al., 2014). Additionally, we provide two stress tests derived from our translation, that deal with semantically equivalent but syntactically different phrasings of the same sentence.

We display some of the evaluation results below. For full details please refer to our paper, which we ask you to cite if you used any of the code, data, or information from the paper:

```
@inproceedings{wijnholds-etal-2021-sicknl,
    title = "SICK-NL: A Dataset for Dutch Natural Language Inference",
    author = "Wijnholds, Gijs and Moortgat, Michael",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2101.05716",
}
```

## Code and results

The code implements the evaluation of English and Dutch BERT/RoBERTa/Multilingual BERT models on SICK and SICK-NL and the two stress tests as Natural Language Inference tasks. As a baseline we also evaluate static embeddings on the relatedness task of SICK and SICK-NL.

We use the [HuggingFace Transformers][transformers] library to load and train the models.
For the Dutch models, we evaluated with [BERTje][bertje] and [RobBERT][robbert].

### Relatedness results (Pearson r)
|                       | SICK  |                       | SICK-NL |
| --------------------- |:-----:| --------------------- |:-------:|
| Skipgram              | 69.49 | Skipgram              |  56.94  |
| BERT<sub>cls</sub>    | 50.78 | BERTje<sub>cls</sub>  |  49.06  |
| BERT<sub>avg</sub>    | 61.36 | BERTje<sub>avg</sub>  |  55.55  |
| RoBERTa<sub>cls</sub> | 46.62 | RobBERT<sub>cls</sub> |  43.93  |
| RoBERTa<sub>avg</sub> | 62.71 | RobBERT<sub>avg</sub> |  52.33  |


### NLI results (threeway classification accuracy)
|         | SICK  |         | SICK-NL |
| ------- |:-----:| ------- |:-------:|
| BERT    | 50.78 | BERTje  |  49.06  |
| mBERT   | 50.78 | mBERT   |  49.06  |
| RoBERTa | 62.71 | RobBERT |  52.33  |


[transformers]: https://github.com/huggingface/transformers
[bertje]: https://github.com/wietsedv/bertje
[robbert]: https://github.com/iPieter/RobBERT
