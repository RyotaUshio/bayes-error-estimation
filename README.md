# Code for the paper "Practical estimation of the optimal classification error with soft labels and calibration"

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

### Appendix B.4: Numerical experiments

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
