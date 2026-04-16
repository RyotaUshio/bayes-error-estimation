# Practical estimation of the optimal classification error with soft labels and calibration

Code for the paper "Practical estimation of the optimal classification error with soft labels and calibration."

Before running experiments, you need to install [uv](https://docs.astral.sh/uv/).

## Experiments

### Section 4.2 & Appendix D.3: Estimation of the Bayes error

To reproduce the experiments presented in Section 4.2 and Appendix D.3, run:

```sh
./scripts/calib_all.sh
```

The result plots will be saved in the `results/calib` directory.

### Section 4.3 & Appendix D.4: Comparing calibration algorithms using FeeBee

To reproduce the experiments presented in Section 4.3 and Appendix D.4, run:

```sh
./scripts/feebee_all.sh
```

To display the results in a tabular format (Table 2), run:

```sh
./scripts/feebee_table.sh
```

### Appendix D: Supplementary for Section 4

#### Appendix D.1: Corruption parameters

To conduct an experiment for a specific set of parameter values $(a, b)$, run, e.g.,

```bash
uv run scripts/calibration/exp.py --dataset synthetic --a 0.4 --b 0.5
```

#### Appendix D.2: Violation of the assumption of Theorem 3

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
