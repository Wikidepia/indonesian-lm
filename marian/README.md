# Neural Machine Translation

> Small brain machine translation experiments.

Model is trained using [Marian NMT](https://github.com/marian-nmt/marian) and available on [Hugging Face](https://huggingface.co/Wikidepia/marian-nmt-enid).


## Model Evaluations (BLEU Score)

Evaluation set is available on [gunnxx/indonesian-mt-data](https://github.com/gunnxx/indonesian-mt-data). There is some overlap train and test set especially with Tatoeba and religious dataset.

#### Valid Set

| Model                      | General | News  | Religious |
| -------------------------- | ------- | ----- | --------- |
| Wikidepia/marian-nmt-enid  | 37.95   | 30.29 | 28.10     |
| Helsinki-NLP/opus-mt-en-id | 32.45   | 26.70 | 24.65     |

#### Test Set

| Model                      | General | News  | Religious |
| -------------------------- | ------- | ----- | --------- |
| Wikidepia/marian-nmt-enid  | 37.41   | 30.85 | 27.80     |
| Helsinki-NLP/opus-mt-en-id | 33.10   | 26.75 | 23.94     |

## TODO

- [x] Train model with OPUS and IndoParaCrawl
- [ ] Client-Side translation [read more](https://browser.mt/)
- [ ] Test [bicleaner](https://github.com/bitextor/bicleaner/) on CCAligned
