# Practical estimation of the optimal classification error with soft labels and calibration

[![](http://img.shields.io/badge/cs.LG-arXiv%3A2505.20761-B31B1B.svg)](https://arxiv.org/abs/2505.20761)

> **Abstract:** 
> While the performance of machine learning systems has experienced significant improvement in recent years, relatively little attention has been paid to the fundamental question: to what extent can we improve our models? This paper provides a means of answering this question in the setting of binary classification, which is practical and theoretically supported. We extend a previous work that utilizes soft labels for estimating the Bayes error, the optimal error rate, in two important ways. First, we theoretically investigate the properties of the bias of the hard-label-based estimator discussed in the original work. We reveal that the decay rate of the bias is adaptive to how well the two class-conditional distributions are separated, and it can decay significantly faster than the previous result suggested as the number of hard labels per instance grows. Second, we tackle a more challenging problem setting: estimation with _corrupted_ soft labels. One might be tempted to use calibrated soft labels instead of clean ones. However, we reveal that _calibration guarantee is not enough_, that is, even perfectly calibrated soft labels can result in a substantially inaccurate estimate. Then, we show that isotonic calibration can provide a statistically consistent estimator under an assumption weaker than that of the previous work. Our method is _instance-free_, i.e., we do not assume access to any input instances. This feature allows it to be adopted in practical scenarios where the instances are not available due to privacy issues. Experiments with synthetic and real-world datasets show the validity of our methods and theory.

Before running experiments, you need to install [uv](https://docs.astral.sh/uv/).

## Experiments

### Section 4

- Fig. 4 (a):
  
  ```bash
  uv run scripts/calibration/exp.py --dataset synthetic --a 2 --b 0.7
  ```
  
- Fig. 4 (b):
  
  ```bash
  uv run scripts/calibration/exp.py --dataset cifar10
  ```

- Fig. 4 (c):
  
  ```bash
  uv run scripts/calibration/exp.py --dataset fashion_mnist
  ```

To visualize the results, use `uv run scripts/calibration/plot.py [RESULT_JSON_FILE]`.

### Appendix D: Supplementary for Section 4

#### Appendix D.2: Corruption parameters

To conduct an experiment for a specific set of parameter values $(a, b)$, run, e.g.,

```bash
uv run scripts/calibration/exp.py --dataset synthetic --a 0.4 --b 0.5
```

#### Appendix D.3: Violation of the assumption of Theorem 2

Run the experiments by:

```bash
bash scripts/calibration/violation_experiment.sh
```

Then, run the following to visualize the results (Fig. 10):

```bash
bash scripts/calibration/violation_plot.sh
```

### Appendix B: Supplementary for Section 2

#### Appendix B.4: Numerical experiments

To reproduce Fig. 7 (a), first run an experiment with:

```bash
uv run scripts/bias/figure7_experiment.py a
```

This will save the result in `results/bias_a.json`. Then run:

```bash
uv run scripts/bias/figure7_plot.py results/bias_a.json
```

to generate a figure from the result.
Fig. 7 (b) and (c) can be obtained similarly.
