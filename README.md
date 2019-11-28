# TonicNet

Accompanying repository for my paper: [Improving Polyphonic Music Models with Feature-Rich Encoding](https://arxiv.org/abs/1911.11775)

<b>Requirements:</b>
- Python 3 (tested with 3.6.5)
- Pytorch (tested with 1.2.0)
- Music21

<b>Prepare Dataset:</b>
```
python main.py --gen_dataset
```

<b>Train Model from Scratch:</b>

First run --gen_dataset, then:
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
