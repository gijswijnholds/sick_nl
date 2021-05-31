# SICK-NL

A translation of the SICK dataset, for evaluating relatedness and entailment models for Dutch. SICK-NL was obtained by semi-automatically translating SICK (Marelli et al., 2014). Additionally, we provide two stress tests derived from our translation, that deal with semantically equivalent but syntactically different phrasings of the same sentence.

We display some of the evaluation results below. For full details please refer to [our EACL 2021 paper], which we ask you to cite if you used any of our code, data, or information from the paper:

```
@inproceedings{wijnholds-etal-2021-sicknl,
    title = "SICK-NL: A Dataset for Dutch Natural Language Inference",
    author = "Wijnholds, Gijs and Moortgat, Michael",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.eacl-main.126/",
}
```

## Code and results

The code implements the evaluation of English and Dutch BERT/RoBERTa/Multilingual BERT models on SICK and SICK-NL and the two stress tests as Natural Language Inference tasks. As a baseline we also evaluate static embeddings on the relatedness task of SICK and SICK-NL.

For relatedness, we use the skipgram vectors of [word2vec][skipgram_vectors], and [Dutch skipgram vectors][dutch_skipgram_vectors]. We use the [HuggingFace Transformers][transformers] library to load and train the models.
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
| BERT    | 87.34 | BERTje  |  83.94  |
| mBERT   | 87.02 | mBERT   |  84.53  |
| RoBERTa | 90.11 | RobBERT |  82.02  |


[skipgram_vectors]: https://code.google.com/archive/p/word2vec/
[dutch_skipgram_vectors]: https://github.com/clips/dutchembeddings
[transformers]: https://github.com/huggingface/transformers
[bertje]: https://github.com/wietsedv/bertje
[robbert]: https://github.com/iPieter/RobBERT
[our EACL 2021 paper]: https://www.aclweb.org/anthology/2021.eacl-main.126/
