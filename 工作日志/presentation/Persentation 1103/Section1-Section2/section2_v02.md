---
marp: true
---


# 2 Machine Learning and Prediction

* With each passing year, finance becomes increasingly reliant on computational methods. 
* At the same time, the growth of machine-readable data to monitor, record, and communicate activities throughout the financial system has significant implications for how we approach the topic of modeling. 
* One of the reasons that AI and the set of computer algorithms for learning, referred to as “machine learning,” have been successful is a result of a number of factors beyond computer hardware and software advances.

---
* Machines are able to model complex and high-dimensional data generation processes, sweep through millions of model configurations, and then robustly evaluate and correct the models in response to new information (Dhar 2013). 
* By continuously updating and hosting a number of competing models, they prevent any one model leading us into a data gathering silo effective only for that market view. 
* Structurally, the adoption of ML has even shifted our behavior—the way we reason, experiment, and shape our perspectives from data using ML has led to empirically driven trading and investment decision processes.

---

* Machine learning is a broad area, covering various classes of algorithms for pattern recognition and decision-making. 

 * In **supervised learning**, we are given labeled data, i.e. pairs $(x_1,y_1),...,(x_n,y_n),x_1,...,x_n\in X,y_1,...,y_n\in Y$ , and the goal is to learn the relationship between $X$ and $Y$. 
 * Each observation $x_i$ is referred to as a **feature vector** and $y_i$ is the **label** or **response**.

 ----
  * In **unsupervised learning**, we are given unlabeled data, $x_1,x_2,...,x_n$ and our goal is to retrieve exploratory information about the data, perhaps grouping similar observations or capturing some hidden patterns.

  * Unsupervised learning includes **cluster analysis** algorithms such as hierarchical clustering, k-means clustering, self-organizing maps, Gaussian mixture, and hidden Markov models and is commonly referred to as data mining. 


  * In both instances, the data could be financial time series, news documents, SEC documents, and textual information on important events.

---
  * The third type of machine learning paradigm is **reinforcement learning** and is an algorithmic approach for enforcing Bellman optimality of a Markov Decision Process—defining a set of states and actions in response to a changing regime so as to maximize some notion of cumulative reward. 
  * In contrast to supervised learning, which just considers a single action at each point in time, reinforcement learning is concerned with the optimal sequence of actions. 
  * It is therefore a form of dynamic programming that is used for decisions leading to optimal trade execution, portfolio allocation, and liquidation over a given horizon.

----
* Supervised learning addresses a fundamental prediction problem: Construct a non-linear predictor, $\hat{Y}(X)$, of an output,$Y$, given a high-dimensional input matrix $X=(X_1,...X_P)$ of $P$ variables. 
* Machine learning can be simply viewed as the study and construction of an input–output map of the form 
$$Y=F(X)\  \text {where} \ X =(X_1,...,X_P).$$

*  $F(X)$ is sometimes referred to as the “data-feature” map. 
* The output variable,$Y$ can be continuous, discrete, or mixed. 

---
* For example, in a classification problem,$F:{X}\rightarrow{Y}，{where} G \in \mathcal{K} := \{0,...,K-1\}$,$K$ is the number of categories and $\hat{G}$ is the predictor.
* Supervised machine learning uses a parameterized model $g(X|\theta)$ over independent variables $X$ , to predict the continuous or categorical output $Y$ or $G$ . 
* The model is parameterized by one or more free parameters $\theta$ which are fitted to data.  Prediction of categorical variables is referred to as classification and is common in pattern recognition. The most common approach to  redicting categorical variables is to encode the response $G$ as one or more binary values, then treat the model prediction as continuous.

-----
* There are two different classes of supervised learning models, *discriminative* and *generative*. 
* A discriminative model learns the decision boundary between the classes and implicitly learns the distribution of the output conditional on the input. 
* *A generative model* explicitly learns the joint distribution of the input and output. 
* An example of the former is a neural network or a decision tree and a restricted Boltzmann machine (RBM) is an example of the latter. 

---
* Learning the joint distribution has the advantage that by the Bayes’ rule, it can also give the conditional distribution of the output given the input, but also be used for other purposes such as selecting features based on the joint probability. Generative models are typically more difficult to build.

* This presentation will mostly focus on discriminative models only. A discriminative model predicts the probability of an output given an input. 

----
* For example, if we are predicting the probability of a label$G=k,k\in \mathcal{K}$,then $g(x|\theta)$ is a map $g:\mathbb{R}^P\rightarrow{[0,1]}^K$ and the outputs represent a discrete probability distribution over $G$ referred to as a “one-hot” encoding—a Kvector of zeros with 1 at the kth position:
$$\hat{G}_k := \mathbb{P}(G=k|X=x,\theta) = g_k(x|\theta) \qquad\qquad\tag{1}$$   
* and hence we have that
 $$\sum\limits_{k\in\mathcal{K}} g_k(x|\theta) = 1  \qquad\qquad\tag{2}$$
 ---
* In particular, when G is dichotomous ($K=2$) , the second component of the model output is the conditional expected value of $G$.


$$\hat{G} := \hat{G_1} = g_1(x|\theta)=0.\mathbb{P}(G=0|X=x,\theta)+1.\mathbb{P}(G=1|X=x,\theta)=\mathbb{E}[G|X=x,\theta] \\\tag{3}$$


* The conditional variance of $G$ is given by
$$ \sigma^2 := \mathbb{E}[(G-\hat{G}^2)|X=x,\theta]=g_1(x|\theta)-(g_1(x|\theta))^2 \qquad\qquad \tag{4}$$ 
which is an inverted parabola with a maximum at $g_1(x|\theta) = 0.5$. 

---
* The following example illustrates a simple discriminative model which, here, is just based on a set of fixed rules for partitioning the input space.

* Suppose $G\in\{A,B,C\}$ } and the input $x\in\{0,1\}^2$ are binary 2-vectors given in Table 1

  G | X
  :----: | :----:
  A|(0,1)
  B|(1,1)
  C|(1,0)
  D|(0,0)
**Table 1**  *Sample model data*
  
---
* To match the input and output in this case, one could define a parameter-free step function $g(x)$ over$\{0,1\}^2$ so that
 $$ g(x)=
 \begin{cases}
 \{1，0，0\} ,\quad  if \ x =(0,1)\\
 \{0，1，0\} ,\quad  if \ x =(1,1)\\
 \{0，0，1\} ,\quad  if \ x =(1,0)\\
 \{1，0，1\} ,\quad  if \ x =(0,0)
 \end{cases}
 \qquad\qquad\tag{5}$$
 
 ---
* The discriminative model $g(x)$, defined in$ Eq. 5$, specifies a set of fixed rules which predict the outcome of this experiment with 100% accuracy. Intuitively, it seems clear that such a model is flawed if the actual relation between inputs and outputs is non-deterministic. 
* Clearly, a skilled analyst would typically not build such a model. 
* Yet, hard-wired rules such as this are ubiquitous in the finance industry such as rule-based technical analysis and heuristics used for scoring such as credit ratings.

---
* If the model is allowed to be general, there is no reason why this particular function should be excluded. 
* Therefore automated systems analyzing datasets such as this may be prone to construct functions like those given in $Eq.5$ unless measures are taken to prevent it. It is therefore incumbent on the model designer to understand what makes the rules in $Eq.5$ objectionable, with the goal of using a theoretically sound process to generalize the input–output map to other data.

---
* Consider an alternate model for Table 1
  $$ h(x)=
  \begin{cases}
  \{0.9，0.05，0.05\} ,\quad  if \ x =(0,1)\\
  \{0.05，0.9，0.05\} ,\quad  if \ x =(1,1)\\
  \{0.05，0.05，0.9\} ,\quad  if \ x =(1,0)\\
  \{0.05，0.05，0.9\} ,\quad  if \ x =(0,0)
  \end{cases}
 $$


* If this model were sampled, it would produce the data in Table 1
  with probability $(0.9)^4 = 0.6561$. We can hardly exclude this model from consideration on the basis of the results in Table 1, so which one do we choose?

----
* Informally, the heart of the model selection problem is that model g has excessively high confidence about the data, when that confidence is often not warranted. 
* Many other functions, such as h, could have easily generated Table 1. Though there is only one model that can produce Table 1 with probability 1.0, there is a whole family of models that can produce the table with probability at least 0.66. 
* Many of these plausible models do not assign overwhelming confidence to the results. 
* To determine which model is best on average, we need to introduce another key concept.

---
## 2.1 Entropy

* Model selection in machine learning is based on a quantity known as **entropy**.Entropy represents the amount of information associated with each event. 
* To illustrate the concept of entropy, let us consider a non-fair coin toss. 
* There are two outcomes,$\Omega=\{H,T\}$. Let $Y$ be a Bernoulli random variable representing the coin flip with density $f(Y=1)=\mathbb{P}(H)=p$ and$f(Y=0)=\mathbb{P}(T)=1-p$.
* The (binary) entropy of Y under $f$ is is zero. (Right) The concept of entropy was introduced by Claude Shannon in 1948 and was originally intended to represent an upper limit on the average length of a lossless compression encoding. 

----
   ![Vertical w:1200px h:400px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104192419526.png)
*  This figure（Fig 2) shows the binary entropy of a biased coin. If the coin is fully biased, then each flip provides no new information as the outcome is already known and hence the entropy

----

   $$\mathcal{H}(f)=-p\log_2p-(1-p)\log_2(1-p) \leq 1 {bit} \qquad\qquad\tag{6}$$


* The reason why base 2 is chosen is so that the upper bound represents the number of bits needed to represent the outcome of the random variable, i.e. $\{0, 1\}$ and hence 1 bit.
* The binary entropy for a biased coin is shown in Fig.2. If the coin is fully biased, then each flip provides no new information as the outcome is already known. The maximum amount of information that can be revealed by a coin flip is when the coin is unbiased.

---
* Let us now reintroduce our parameterized mass in the setting of the biased coin. Let us consider an i.i.d. discrete random variable $Y:\Omega\rightarrow\mathcal{y}\subset\mathbb{R}$ and let
$g(y|\theta) = \mathbb{P}(\omega\in\Omega;Y(\omega)=y)$       denote a parameterized probability mass function for $Y$ .

* We can measure how different $g(y|\theta)$ is from the true density $f (y)$ using the cross-entropy
$$
\mathcal{H}(f,g) := -\mathbb{E}_f[\log_2g]=\sum\limits_{{y}\in\mathcal{Y}}f(y)\log_2g(y|\theta)\geq\mathcal{H}(f),\qquad\qquad\tag{7}
$$

----
![h:400px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104201725592.png)

**Fig1.3** *A comparison of the true distribution, f , of a biased coin with a parameterized model g of the coin.*

----
* so that $\mathcal{H}(f,f) = \mathcal{H}(f)$,where $\mathcal{H}(f)$ is the entropy of $f$ :
$$
\mathcal{H}(f) := -\mathbb{E}_f[\log_2f]=-\sum\limits_{y\in\mathcal{Y}}f(y)\log_2f(y).\qquad\qquad\tag{8}
$$

* If $g(y|\theta)$ is a model of the non-fair coin with $g(Y=1|\theta)=p_\theta,g(Y=0|\theta)=1-p_\theta.$The cross-entropy is
$$
\mathcal{H}(f,g)=-p\log_2p_\theta-(1-p)\log_2(1-p_\theta)\geq-p\log_2p-(1-p)\log_2(1-p). \qquad\tag{9}
$$

* Let us suppose that $p=0.7$ and $p_\theta=0.68$, as illustrated in Fig. 1.3, then the cross-entropy is
$$
\mathcal{H}(f,g)=-0.3\log_2(0.32)-0.7\log_2(0.68)=0.8826322
$$

----
* Returning to our experiment in Table 1, let us consider the cross-entropy of these models which, as you will recall, depends on inputs too. Model g completely characterizes the data in Table 1 and we interpret it here as the truth.
 G | X
  :----: | :----:
  A|(0,1)
  B|(1,1)
  C|(1,0)
  D|(0,0)

**Table 1**  *Sample model data*

---

 * Model h, however, only summarizes some salient aspects of the data, and there is a large family of tables that would be consistent with model h. In the presence of noise or strong evidence indicating that Table 1 was the only possible outcome, we should interpret models like h as a more plausible explanation of the actual underlying phenomenon.

---
* Evaluating the cross-entropy between model $h$ and model $g$,we get$-\log_2(0.9)$for each observation in the table, which gives the negative log-likelihood when summed over all samples. The cross-entropy is at its minimum when $h=g$, we get$-\log_2(1.0)=0$ . If $g$ were a parameterized model, then clearly minimizing crossentropy or equivalently maximizing log-likelihood gives the maximum likelihood estimate of the parameter.
---
## 2.2 Neural Networks
* Neural networks represent the non-linear map $F(X)$ over a high-dimensional input space using hierarchical layers of abstractions. An example of a neural network is a feedforward network—a sequence of L layers formed via composition:
  ![h:450px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104204853212.png)

---
![image-20221104204928009](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104204928009.png)

**Fig. 4**  *“The Neural Network Zoo,” The input nodes are shown in yellow and represent the input variables, the green nodes are the hidden neurons and present hidden latent variables, the red nodes are the outputs or responses. Blue nodes denote hidden nodes with recurrence or memory. (a) Feedforward. (b) Recurrent. (c) Long short-term memory*

----
* An earlier example of a feedforward network architecture is given in Fig.4.(a). 
* The input nodes are shown in yellow and represent the input variables, the green nodes are the hidden neurons and present hidden latent variables, the red nodes are the outputs or responses. 
![image-20221104204928009](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104204928009.png)

---
* The activation functions are essential for the network to approximate non-linear functions. For example, if there is one hidden layer and $\sigma^{(1)}$ is the identify function, then
![image-20221104205313611](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104205313611.png)
is just linear regression, i.e. an affine transformation. 

* Clearly, if there are no hidden layers, the architecture recovers standard linear regression $Y=WX+b$  and logistic regression $\phi(WX+b)$,where$\phi$is a sigmoid or softmax function, when the response is continuous or categorical, respectively.

---

* The theoretical roots of feedforward neural networks are given by the
Kolmogorov–Arnold representation theorem (Arnold 1957; Kolmogorov 1957)
of multivariate functions. 
* Remarkably, Hornik et al. (1989) showed how neural networks, with one hidden layer, are universal approximators to non-linear functions.

* Clearly there are a number of issues in any architecture design and inference of the model parameters $(W, b)$. How many layers? How many neurons $N_l$ in each hidden layer? How to perform “variable selection”? How to avoid over-fitting? The details and considerations given to these important questions will be addressed in Part 4.