---
marp: true
---

# 6 Summary
* In this presentation, we have identified some of the key elements of supervised machine
learning. Supervised machine learning
1. is an algorithmic approach to statistical inference which, crucially, does not
depend on a data generation process;
2. estimates a parameterized map between inputs and outputs, with the functional
form defined by the methodology such as a neural network or a random forest;
3. automates model selection, using regularization and ensemble averaging techniques to iterate through possible models and arrive at a model with the best
out-of-sample performance; and
4. is often well suited to large sample sizes of high-dimensional non-linear covariates.

* The emphasis on out-of-sample performance, automated model selection, and absence of a pre-determined parametric data generation process is really the key to machine learning being a more robust approach than many parametric, financial econometrics techniques in use today. The key to adoption of machine learning in finance is the ability to run machine learners alongside their parametric counterparts, observing over time the differences and limitations of parametric modeling based on in-sample fitting metrics. Statistical tests must be used to characterize the data and guide the choice of algorithm, such as, for example, tests for stationary. See Dixon and Halperin (2019) for a checklist and brief but rounded discussion of some of the challenges in adopting machine learning in the finance industry.

* Capacity to readily exploit a wide form of data is their other advantage, but only if that data is sufficiently high quality and adds a new source of information.  We close this presenation with a reminder of the failings of forecasting models during the financial crisis of 2008 and emphasize the importance of avoiding siloed data extraction.  The
application of machine learning requires strong scientific reasoning skills and is not a panacea for commoditized and automated ecision-making.
