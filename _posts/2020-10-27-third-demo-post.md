---
layout: post
title: Clinical Trial Research
date: 2020-10-27 13:32:20 +0300
description: Clinical Trial Research using data from different studies, also needed to generate Synthetic data from Real data for Clinical Researchers
image: /assets/images/posts/ClinicalTrial.png
fig-caption: # Add figcaption (optional)
tags: [Python, MultiVariate TimeSeries, NLP, RNN, CNN, TensorFlow 2, TimeGAN, CGAN]
---

 Designed and Developed Clinical Trail management system for Clinical Researchers by analyzing different studies conducted across different parts of the world, real time data 

 for certain trials were less and to increase quantity of real time data using TimeGAN synthetic data was generated which has all the parameters similar to real time data.

 1. Data for processing is been received from Web API in the form of JSON and Python 3.8 was used to process data.

 2. With the available data joint training of an autoencoder and adversarial network.

 3. TimeGAN was implemented using TensorFlow 2 by preparing real and random input series by using SciKit MinMaxScalar to rescale the synthetic data.

 4. Autoencoder integrates embedder and recovery networks that is being trainned with real data.

 5. Supervised learning with real time data.

 6. Joint training with real and random data.

 7. Generating Time Series data and evaluating the quality of synthetic time series data using PCA and t-SNE (Visualization).

 8. Evaluating fidelity based on time-series classification.

 9. Assessing usefulness – train on synthetic and test on real.

 


