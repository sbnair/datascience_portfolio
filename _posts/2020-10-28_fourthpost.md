---
layout: post
title: Real time Stock Analysis
date: 2020-10-28 13:32:20 +0300
description: Analysing the dataset using Apache Spark, Kafka, Druid.
image: /assets/images/posts/RealStreaming.jpeg
fig-caption: # Add figcaption (optional)
tags: [Apache Spark, Kafka, Druid, C#, Java, ZedGraph charts]
---

 In this project, Would like to share a simple example of how using open source technologies can we develop a robust distributed computing system that can compute and process large amount's of data in real time and show user the various insights predicted without any bottlenecks, in this below diagram i have used tick by tick data of different stocks to compute and display financial indicators in chart but this architecture can be used in any business domain.

 1. Fetch data from IQFeed API (tick by tick data), to fetch data from IQfeed api cron jobs monitored by apache airflow has been used to do the scheduled fetching of tick by tick data.

 2. Process Raw tick by tick data into different topics processed by Kafka using C#.net and also for some stocks done by Java to prevent losses.

 3. Apache Kafka produces data which is consumed and processed by Apache Spark (using Scala) and the data is stored in Apache NiFi and partly in MongoDB.

 4. Apache Spark processes the images and unstructured data using CNN and RNN Nueral Networks.

 5. Apache Kafka produces data in two modes, one is the windowed and other one is real time streaming which can be queried using Adhoc queries in Apache Druid.

 6. Apache Superset is used to display reports in a very visual friendly way to the end customer which contains both Windowed data and real streaming data.     

