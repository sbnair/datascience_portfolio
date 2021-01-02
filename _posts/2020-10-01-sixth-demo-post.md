---
layout: post
title: IoT Data Analytics
date: 2021-01-02 13:32:20 +0300
description: Data Analytics using IoT Sensors 
image: /assets/images/posts/GolangML.png
fig-caption: # Add figcaption (optional)
tags: [Scala, Golang, GoML, Postgress DB, Apache Spark, Apache Kafka, PMML, Goraph, Javascript, ReactJS]
---

 Designed and Developed IoT Analytics system which collects data from various data sources, analyzes and plots dat into Goraph charts: 

 1. IoT sensors from various locations generates data and is send to server in the form of raw format.

 2. Converter which is written in java language is used to convert Raw Data into XML format.

 3. Real Time streaming of data from IoT sensors is done using Apache Kafka which establishes Producer and Consumer, Consumer consumes data from Kafka and process the data using Apache Spark, the busines logics are written in Scala.

 4. Rules are defined in DB which are read and stored in PMML (Predictive Mark up language) format and it is referred to convert data retrieved from Kafka Consumer.

 5. Based on the XML Data and PMML rules machine learning models are created using GoML package, process are controlled concurrently using Go Routines in the form of channels. 

 6. Gorgonia library is used to manage Nueral Networks and execute using Theano server library.
 
 7. Results generated using Gorgonia are displayed in Web using ReactJS, Goraph and the charts are downloaded in svg format using SVGo.
 
 

 


