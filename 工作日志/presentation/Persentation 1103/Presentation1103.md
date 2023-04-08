---
marp: true
---
# 4 Reinforcement Learning

---
* Recall that supervised learning is essentially a paradigm for inferring the parameters
of a map between input data and an output through minimizing an error over training
samples. Performance generalization is achieved through estimating regularization
parameters on cross-validation data. Once the weights of a network are learned,
they are not updated in response to new data. For this reason, supervised learning
can be considered as an “offline” form of learning, i.e. the model is fitted offline.
Note that we avoid referring to the model as static since it is possible, under certain
types of architectures, to create a dynamical model in which the map between input
and output changes over time. For example, as we shall see in Chap. 8, a LSTM
maintains a set of hidden state variables which result in a different form of the map
over time.

----
* In such learning, a “teacher” provides an exact right output for each data point
in a training set. This can be viewed as “feedback” from the teacher, which for
supervised learning amounts to informing the agent with the correct label each time
the agent classifies a new data point in the training dataset. Note that this is opposite
to unsupervised learning, where there is no teacher to provide correct answers to a
ML algorithm, which can be viewed as a setting with no teacher, and, respectively,
no feedback from a teacher.

----
* An alternative learning paradigm, referred to as “reinforcement learning,” exists
which models a sequence of decisions over state space. The key difference of
this setting from supervised learning is feedback from the teacher is somewhat
in between of the two extremes of unsupervised learning (no feedback at all) and
supervised learning that can be viewed as feedback by providing the right labels.
Instead, such partial feedback is provided by “rewards” which encourage a desired
behavior, but without explicitly instructing the agent what exactly it should do, as in
supervised learning.

---
* The simplest way to reason about reinforcement learning is to consider machine
learning tasks as a problem of an agent interacting with an environment, as
illustrated in Fig 6.

![fig6.png](Presentation1104\fig6.png)


