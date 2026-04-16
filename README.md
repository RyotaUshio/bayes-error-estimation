# Practical estimation of the optimal classification error with soft labels and calibration

<!-- begin-unname -->

[![](http://img.shields.io/badge/cs.LG-arXiv%3A2505.20761-B31B1B.svg)](https://arxiv.org/abs/2505.20761)

<!-- end-unname -->

This repository contains:

- [packages/bestperf](packages/bestperf): Python library that implements the method proposed in the paper "Practical estimation of the optimal classification error with soft labels and calibration."
- [packages/bayes-error-paper](packages/bayes-error-paper): Code for reproduction of the results presented in the paper.

> **Abstract:**
> While the performance of machine learning systems has experienced significant improvement in recent years, relatively little attention has been paid to the fundamental question: to what extent can we improve our models? This paper provides a means of answering this question in the setting of binary classification, which is practical and theoretically supported. We extend a previous work that utilizes soft labels for estimating the Bayes error, the optimal error rate, in two important ways. First, we theoretically investigate the properties of the bias of the hard-label-based estimator discussed in the original work. We reveal that the decay rate of the bias is adaptive to how well the two class-conditional distributions are separated, and it can decay significantly faster than the previous result suggested as the number of hard labels per instance grows. Second, we tackle a more challenging problem setting: estimation with *corrupted* soft labels. One might be tempted to use calibrated soft labels instead of clean ones. However, we reveal that *calibration guarantee is not enough*, that is, even perfectly calibrated soft labels can result in a substantially inaccurate estimate. Then, we show that isotonic calibration can provide a statistically consistent estimator under an assumption weaker than that of the previous work. Our method is *instance-free*, i.e., we do not assume access to any input instances. This feature allows it to be adopted in practical scenarios where the instances are not available due to privacy issues. Experiments with synthetic and real-world datasets show the validity of our methods and theory.
