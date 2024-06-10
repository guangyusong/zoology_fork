## Notes:
* This only works in Python 3.10 because of ray support.

nohup time python -m zoology.launch zoology/experiments/arxiv24_based_figure2/configs.py -p > output.log 2>&1 &

python -m zoology.launch zoology/experiments/arxiv24_based_figure2/configs.py -p

To create plots:
```bash
python zoology/experiments/arxiv24_based_figure2/plot.py
```

## Errors:

Python.h missing

```bash
sudo apt-get update
sudo apt-get install python3.10-dev
```

For Mamba to work

```bash
pip install wheel
pip install mamba-ssm==1.2.0.post1
pip install causal_conv1d==1.1.1
```


Mamba:
python -m zoology.launch zoology/experiments/examples/mamba.py