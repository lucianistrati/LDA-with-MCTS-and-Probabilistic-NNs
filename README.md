# LDA-with-MCTS-and-Probabilistic-NNs

In natural language processing, Latent Dirichlet Allocation (LDA) is a generative statistical model that explains a set of observations through unobserved groups, and each group explains why some parts of the data are similar. LDA is an example of a topic model. In this model, observations (e.g., words) are collected into documents, and each word's presence is attributable to one of the document's topics. Each document will contain a small number of topics.

## Introduction
This repository implements a combination of Latent Dirichlet Allocation (LDA) with Monte Carlo Tree Search (MCTS) and Probabilistic Neural Networks (PNNs) for various tasks such as binary and multiclass classification. The main goal is to explore the synergies between these techniques in natural language processing and classification tasks.

## Contents
- `main.py`: Contains Python code for implementing binary and multiclass classification using both normal and Bayesian neural networks, alongside visualization functions and dataset generation.
- `lda_comment.py`: Implements Latent Dirichlet Allocation (LDA) for topic modeling on textual data using Markov Chain Monte Carlo (MCMC) sampling. It includes functionalities for fitting LDA models to input data and analyzing the results.

## Dependencies
- `numpy`
- `scipy`
- `matplotlib`
- `tensorflow`
- `tensorflow_probability`
- `scikit-learn`
- `pymc`

## Usage
1. Ensure all dependencies are installed in your environment.
2. Run `main.py` to execute binary and multiclass classification tasks using neural networks.
3. Run `lda_comment.py` to perform topic modeling with LDA on textual data.

## Examples
- `main.py` includes examples of binary and multiclass classification tasks using both normal and Bayesian neural networks.
- `lda_comment.py` provides examples of LDA topic modeling on sample textual data.


[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://stand-with-ukraine.pp.ua)
