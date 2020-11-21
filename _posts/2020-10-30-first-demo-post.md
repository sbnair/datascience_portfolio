---
layout: post
title: Analyzing Trading patterns based on different rules stored in DB.
date: 2020-10-30 13:32:20 +0300
description: Using CNN and LSTM trading patterns are analyzed based on different rules.
image: /assets/images/posts/ChartRecognition.png
fig-caption: # Add figcaption (optional)
tags: [Python, MongoDB, OpenCV C++, CNN, LSTM]
---

 Designed and Developed Pattern matching analyzer using Python which reads multiple images uploaded in server to detect patterns based on the rules defined by Quants in MongoDB.

 1. Input images uploaded in server need to have OHLC format with proper labels for Price and Time labels in X and Y axis.

 2. OHLC Format should be of 1 minute timeframe which is used to detect patters based on rules stored.

 3. DTV (Dynamic Time Wrapper) is used to identify patterns using Open CV 4 C++ library.

 4. Hardcoded recgonizer is used to recognize bearish and bullish flags in chart.

 5. Based on recognizer training and testing set is buld by 90/10 and 80/20 ratio.

 6. Training data set is done using 2D CNN and LSTM using RNN in Pytorch library.

 7. Images are read using 2D CNN and the textual data retrieved is added into LSTM for more input processing.

 8. If a data which is read matches the rules defined in DB like RSI pattern, ARIMA model, etc then an email is send to user by adding the details of matching with the image attached into it.





