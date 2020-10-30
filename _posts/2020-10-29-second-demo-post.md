---
layout: post
title: Analyzing Sentimental Analysis using RNN, CNN for capturing fuel statistics and flight performance based on Text and Image based Data
date: 2020-10-29 13:32:20 +0300
description: Text Data being captured from different sources which is being used by RNN and CNN nueral networks to process and display reports
image: /assets/images/posts/RealStreaming.jpeg
fig-caption: # Add figcaption (optional)
tags: [Python, MultiVariate TimeSeries, NLP, RNN, CNN, TensorFlow 2]
---

 Designed and Developed FOQA readout analysis product which is being used by pilots, flight data analysts in different airlines, during accidents data from BlackBox Recorders are taken and processed which comes in the form of .bin,.dat,.fdt formats and also Imagery datasets taken from different GIS products to analyze.RNN and CNN along with other techs were used to do investigation of accidents and also analyze Pilot behaviour, fuel usage efficiency, predict Fuel usage and Aircraft parts maintenance or replacement.

 1. Flight data from blackbox recorders, SAGE, etc for different aircraft types were taken and grouped based on Pilots to analyze timeseries data.

 2. Flight paths were received from ArcGIS and openflights.org to analyze different flight routes.

 3. Flights review were taken from different social networks to get and understand the sentiments of the flights based on comments from the customers.

 4. All the data sets from different sources are bundled into TensorFlow 2 to process and analyze.

 5. Defined the Domain Specific Word Embeddings based on business logic, Recurrent drop out for regularization has been used, Word Vectors trainned on large data.

 6. TensorFlow provides a Tokenizer to convert text docs into integer encoded sequences and Defined the RNN archtiecture with forezen weights across all the nueral networks used.

 7. Preparing Data for RNN model to predict, we created 90:10 training/test split model and also used the Bidirectional GRU unit that scans the text both forward and backward.

 8. Compiled using AdamOptimizer and targetted the Mean Sqaured Loss for this task and we trainned for upto 100 epochs and also tested this model.

 9. Combined the text data and images that we got to use stackedLSTM to process multiple inputs along with CNN to get data.

 10. RNNs captured long range dependencies to capture fuel usage predict future fuel usage and predict maintenance if different aircraft types.

 11. CNNs captured different images to follow the flight plan, climate, locations , etc using object detection, transfer learning which uses pretrained weights to do the processing using Keras in Python.



