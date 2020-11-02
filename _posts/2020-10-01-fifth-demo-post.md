---
layout: post
title: Renewable Energy Distribution
date: 2020-10-01 13:32:20 +0300
description: Renewable Energy Distribution across different areas 
image: /assets/images/posts/reneweableenergygrid.png
fig-caption: # Add figcaption (optional)
tags: [Python, pymdptoolbox, Q-Learning, MDP, Web API, pyplot, keras, scikit-learn]
---

 Designed and Developed Deep Reinforcement Learning platform to solve demand-supply deficit in Renewable Energy industry to lower energy waistage and control energy production costs : 

 1. Data regarding consumer and producer meter data was recieved in the form of certifcates from different sources based in Europe like Elhub, Statnett (NECS), etc.

 2. Data was stored in the form of HDS so that it is easy to access from Python.

 3. MDP (Markov Decision Process) was used to define rewards to the reinforcement agent and find policies to optimize on the go based on state of value functions.

 4. MDPs problem including Bellman Equations are solved using pymdptoolbox library using Q-Learning Algo inorder to find a optimal policy based on the data or state recieved 

 from different consumers, it takes both negative and postive rewards and finds an optimal policy to move away from negative rewards.

 5. Epsilon-Greedy Policy is used in Q-Learning which does Exploitation of data and also exploration by learning past data to optimize rewards.

 6. Experience replay was used which focus on past mistakes by storing states, rewards and next states in memory in simple batches to learn based on past mistakes.
 
 7. Double Deep Q Learning was used since Q-Learning caused approximate values rather than accurate values in states.
 
 8. Double Deep Q Learning algorithm uses the weights of one network to select the best action given the next state and also as well as the weights of another network.


 


