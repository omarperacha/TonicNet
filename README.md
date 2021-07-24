# TonicNet

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/js-fake-chorales-a-synthetic-dataset-of/music-modeling-on-jsb-chorales)](https://paperswithcode.com/sota/music-modeling-on-jsb-chorales?p=js-fake-chorales-a-synthetic-dataset-of-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/improving-polyphonic-music-models-with/music-modeling-on-jsb-chorales)](https://paperswithcode.com/sota/music-modeling-on-jsb-chorales?p=improving-polyphonic-music-models-with)


Accompanying repository for my paper: [Improving Polyphonic Music Models with Feature-Rich Encoding](https://arxiv.org/abs/1911.11775)

<b>Requirements:</b>
- Python 3 (tested with 3.6.5)
- Pytorch (tested with 1.2.0)
- Music21

<b>Prepare Dataset:</b>

To prepare the vanilla JSB Chorales dataset with canonical train/validation/test split:
```
python main.py --gen_dataset
```

To prepare dataset augmented with [JS Fake Chorales](https://github.com/omarperacha/js-fakes):
```
python main.py --gen_dataset --jsf
```

To prepare dataset for training on JS Fake Chorales only:
```
python main.py --gen_dataset --jsf_only
```

<b>Train Model from Scratch:</b>

First run `--gen_dataset` with any optional 2nd argument, then:
```
python main.py --train
```

Training requires 60 epochs, taking roughly 3-6 hours on GPU

<b>Evaluate Pre-trained Model on Test Set:</b>

```
python main.py --eval_nn
```

<b>Sample with Pre-trained Model (via random sampling):</b>

```
python main.py --sample
```
