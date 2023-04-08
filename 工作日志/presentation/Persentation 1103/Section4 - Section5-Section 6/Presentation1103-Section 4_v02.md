---
marp: true
---
# 4 Reinforcement Learning

---
* Recall that supervised learning is essentially a paradigm for inferring the parameters of a map between input data and an output through minimizing an error over training samples. 
* Performance generalization is achieved through estimating regularization parameters on cross-validation data. 
* Once the weights of a network are learned, they are not updated in response to new data. 
* For this reason, supervised learning can be considered as an “offline” form of learning, i.e. the model is fitted offline.
* Note that we avoid referring to the model as static since it is possible, under certain types of architectures, to create a dynamical model in which the map between input
and output changes over time. 

---
* In such learning, a “teacher” provides an exact right output for each data point
in a training set. This can be viewed as “feedback” from the teacher, which for
supervised learning amounts to informing the agent with the correct label each time
the agent classifies a new data point in the training dataset. Note that this is opposite to unsupervised learning, where there is no teacher to provide correct answers to a ML algorithm, which can be viewed as a setting with no teacher, and, respectively, no feedback from a teacher.

---
* An alternative learning paradigm, referred to as **“reinforcement learning,”** exists which models a sequence of decisions over state space. 
* The key difference of this setting from supervised learning is feedback from the teacher is somewhat in between of the two extremes of unsupervised learning (no feedback at all) and supervised learning that can be viewed as feedback by providing the right labels.
* Instead, such partial feedback is provided by “rewards” which encourage a desired behavior, but without explicitly instructing the agent what exactly it should do, as in supervised learning.

---
* The simplest way to reason about reinforcement learning is to consider machine learning tasks as a problem of an agent interacting with an environment, as illustrated in Fig6.

![width:500px](Presentation1104\fig6.png)


**Fig.6** *This figure shows a reinforcement learning agent which performs actions at times $t_0,...,t_n$. The agent perceives the environment through the state variable $S_t$ . In order to perform better on its task, feedback on an action $a_t$ is provided to the agent at the next time step in the form of a reward $R_t$*

---

* The agent learns about the environment in order to perform better on its task, which can be formulated as the problem of performing an optimal action. 
* If an action performed by an agent is always the same and does not impact the environment, in this case we simply have a perception task, because learning about the environment helps to improve performance on this task. 

---
* For example, you might have a model for prediction of mortgage defaults where the action is to compute the default probability for a given mortgage. The agent, in this case, is just a predictive model that produces a number and there is measurement of how the model impacts the environment. 
* For example, if a model at a large mortgage broker predicted that all borrowers will default, it is very likely that this would have an impact on the mortgage market, and consequently future predictions. 
* However, this feedback is ignored as the agent just performs perception tasks, ideally suited for supervised learning. Another example is in trading. 
* Once an action is taken by the strategy there is feedback from the market which is referred to as “market impact.”

---
* Such a learner is configured to maximize a long-run utility function under
some assumptions about the environment. 
* One simple assumption is to treat the environment as being fully observable and evolving as a first-order Markov process.

* A Markov Decision Process (MDP) is then the simplest modeling framework that allows us to formalize the problem of reinforcement learning. A task solved by MDPs is the problem of **optimal control**, which is the problem of choosing action variables over some period of time, in order to maximize some objective function that depends both on the future states and action taken. 

---
* In a discrete-time setting, the state of the environment $S_t ∈ S$ is used by the learner (a.k.a. agent) to decide which action $a_t ∈ A(S_t)$ to take at each time step. 
* This decision is made dynamic by updating the probabilities of selecting each action conditioned on St .
* These conditional probabilities $π_t(a|s)$ are referred to as the agent’s **policy**. 
* The mechanism for updating the policy as a result of its learning is as follows: one time step later and as a consequence of its action, the learner receives a reward defined by a **reward function**, an immediate reward given the current state St and action taken $a_t$ .

----

* As a result of the dynamic environment and the action of the agent, we transition
to a new state $S_{t+1}$. 
* A reinforcement learning method specifies how to change the policy so as to maximize the total amount of reward received over the long-run. 
* The structure of reinforcement learning will not be formally elaborated here, we will only discuss informally some of the challenges of reinforcement learning in finance.

---
* Most of the impressive progress reported recently with reinforcement learning
by researchers and companies such as Google’s DeepMind or OpenAI, such as
playing video games, walking robots, self-driving cars, etc., assumes complete
observability, using Markovian dynamics. 
* The much more challenging problem, which is a better setting for finance, is how to formulate reinforcement learning for partially observable environments, where one or more variables are hidden.

---
* Another, more modest, challenge exists in how to choose the optimal policy when no environment is fully observable but the dynamic process for how the states evolve over time is unknown. 
* It may be possible, for simple problems, to reason about how the states evolve, perhaps adding constraints on the state-action space. However,the problem is especially acute in high-dimensional discrete state spaces, arising from, say, discretizing continuous state spaces. 
* Here, it is typically intractable to enumerate all combinations of states and actions and it is hence not possible to exactly solve the optimal control problem. 
* In particular, we will turn to neural networks to approximate an action function known as a “Q-function.” 
* Such an approach is referred to as “Q-Learning” and more recently, with the use of deep learning to approximate the Q-function, is referred to as “Deep Q-Learning.”

---

* To fix ideas, we consider a number of examples to illustrate different aspects of the problem formulation and challenge in applying reinforcement learning. We start with arguably the most famous toy problem used to study stochastic optimal control theory, the “**multi-armed bandit problem**.” This problem is especially helpful in developing our intuition of how an agent must balance the competing goals of exploring different actions versus exploitation of known outcomes.

---

### Example 3 Multi-armed Bandit Problem
* Suppose there is a fixed and finite set of n actions, a.k.a. arms, denoted $A$. Learning proceeds in rounds, indexed by $t = 1,...,T$ . The number of rounds $T $, a.k.a. the time horizon, is fixed and known in advance. In each round, the agent picks an arm at and observes the reward $R_t(a_t)$ for the chosen arm only. For avoidance of doubt, the agent does not observe rewards for other actions that could have been selected. 
* If the goal is to maximize total reward over all rounds, how should the agent choose an arm?

---
* Suppose the rewards Rt are independent and identical random variables with
distribution $ν ∈ [0, 1]^
n$ and mean $μ$. The best action is then the distribution
with the maximum mean $μ^∗$.

* The difference between the player’s accumulated reward and the maximum
the player (a.k.a. the “cumulative regret”) could have obtained had she known
all the parameters is

$$
\bar{R}_T = T\mu^* - \mathbb E \sum_{t\in[T]}R_t
$$

* Intuitively, an agent should pick arms that performed well in the past, yet
the agent needs to ensure that no good option has been missed.

---

* The theoretical origins of reinforcement learning are in stochastic dynamic
programming. In this setting, an agent must make a sequence of decisions under
uncertainty about the reward. 
* If we can characterize this uncertainty with probability distributions, then the problem is typically much easier to solve. We shall assume that you has some familiarity with dynamic programming—the extension to stochastic dynamic programming is a relatively minor conceptual development.
* The following optimal payoff example will    As we follow the mechanics of solving the problem, the example exposes the inherent difficulty of relaxing our assumptions about the distribution of the uncertainty.

---

### Example 4 Uncertain Payoffs

* A strategy seeks to allocate \$600 across 3 markets and is equally profitable
once the position is held, returning 1% of the size of the position over a short
trading horizon $[t,t + 1]$. However, the markets vary in liquidity and there is
a lower probability that the larger orders will be filled over the horizon. The
amount allocated to each market must be either $K = \{100, 200, 300\}$.

![width:800px](Presentation1104\fig7.png)

----

* The optimal allocation problem under uncertainty is a stochastic dynamic programming problem. 
* We can define value functions vi(x) for total allocation amount x for each stage of the problem, corresponding to the markets. 
* We then find the optimal allocation using the backward recursive formulae:
$$
v_3(x)=R_3,\forall x \in K\\
\qquad
v_2(x) = \underset{k \in K}{max}{\{R_2(k)+v_3(x-k)\}},\forall x \in K+200,\\

\qquad
v_1(x) = \underset{k \in K}{max}{\{R_1(k)+v_2(x-k)\}},x = 600,
$$


* The left-hand side of the table below tabulates the values of $R_2 + v_3$
corresponding to the second stage of the backward induction for each pair
$(M_2, M_3)$.

---

![width:1200px](Presentation1104\fig8.png)

* The right-hand side of the above table tabulates the values of $R_1 + v_2$ corresponding to the third and final stage of the backward induction for each tuple $(M_1, M_2^∗, M_3^∗)$.

---
* In the above example, we can see that the allocation $\{200, 200, 200\}$ maximizes $v_1(600) = 4.3$.   
* While this eample is a straightforward application of a Bellman optimality recurrence relation, it provides a glimpse of the types of stochastic dynamic programming problems that can be solved with reinforcement learning.   
* In particular, if the fill probabilities are unknown but must be learned over time by observing the outcome over each period $[t_i, t_i+1)$, then the problem above cannot be solved by just using backward recursion.  

---
* Instead we will move to the framework of reinforcement learning and attempt to learn the best actions given the data.   
* Clearly, in practice, the example is much too simple to be representative of real-world problems in finance—the profits will be unknown and the state space is significantly larger, compounding the need for reinforcement learning.   
* However, it is often very useful to benchmark reinforcement learning on simple stochastic dynamic programming problems with closed-form solutions.

---

* In the previous example, we assumed that the problem was static—the variables
in the problem did not change over time. 
* This is the so-called static allocation problem and is somewhat idealized. 
* Our next example will provide a glimpse of the types of problems that typically arise in optimal portfolio investment where random variables are dynamic. 
* The example is also seated in more classical finance theory, that of a “Markowitz portfolio” in which the investor seeks to maximize a risk-adjusted long-term return and the wealth process is self-financing.

---
### Example 5 Optimal Investment in an Index Portfolio

* Let St be a time-t price of a risky asset such as a sector exchange-traded fund
(ETF). We assume that our setting is discrete time, and we denote different time
steps by integer valued-indices $t = 0,...,T$ , so there are $T + 1$ values on our
discrete-time grid. The discrete-time random evolution of the risky asset $S_t$ is
$$
S_{t+1} = S_t(1+\phi_t)\qquad\qquad\tag{12}
$$

---
$$
S_{t+1} = S_t(1+\phi_t)\qquad\qquad\tag{12}
$$

* where $\phi_t$ is a random variable whose probability distribution may depend on
the current asset value $S_t$ . To ensure non-negativity of prices, we assume that
$\phi_t$ is bounded from below $\phi_t ≥ −1$.

---
* Consider a wealth process $W_t$ of an investor who starts with an initial wealth
$W_0$ = 1 at time $t = 0$ and, at each period $t = 0,...,T − 1$ allocates a fraction
$u_t = u_t(S_t)$ of the total portfolio value to the risky asset, and the remaining
fraction $1 − u_t$ is invested in a risk-free bank account that pays a risk-free
interest rate $r_f = 0$. 
* We will refer to a set of decision variables for all time steps as a policy $\pi:= \{u_t\}^{T-1}_{t=0}$. The wealth process is self-financing and so the
wealth at time $t + 1$ is given by

$$
W_{t+1} =(1-u_t)W_t+u_tW_t(1+\phi_t)\qquad\qquad\tag{13}
$$

This produces the one-step return

$$
r_t={ {W_{t+1}-W_t}\over{W_t} }= u_t{\phi_t}\qquad\qquad\tag{14}
$$

---
* Note this is a random function of the asset price $S_t$ . We define one-step rewards
$R_t$ for $t = 0,...,T − 1$ as risk-adjusted returns

$$
R_t = r_t-\lambda Var[r_t|S_t] = u_t{\phi}_t-\lambda u^2Var[{\phi}_t|S_t],\qquad\qquad\tag{15}
$$

where $λ$ is a risk-aversion parameter.We now consider the problem of
maximization of the following concave function of the control variable $u_t$ :
$$
V^{\pi}(s) = \underset{u_t}{max}\mathbb E\left[\sum_{t_0}^T{R_t}|S_t = s \right] = \underset{u_t}{max}\mathbb E \left[ \sum_{t=0}^T u_t \phi _t -\lambda {u_t}^2Var[\phi _t|S_t]\middle|S_t = s \right] \qquad\tag{16}
$$

* This equation defines an optimal investment problem for T − 1 steps faced
by an investor whose objective is to optimize risk-adjusted returns over each period. 

----
* This optimization problem is equivalent to maximizing the long-run
 returns over the period $[0, T]$. For each $t = T −1, T −2,..., 0,$ the optimality
condition for action ut is now obtained by maximization of $V^π (s)$ with respect
to $u_t$ :

$$
u^* = {{{\mathbb{} E }[\phi _t|S_t]}\over{2 \lambda Var[\phi _t|S_t]}}\qquad\tag{17}
$$

* where we allow for short selling in the ETF $(i.e., u_t < 0)$ and borrowing of
cash $u_t > 1$.

* This is an example of a stochastic optimal control problem for a portfolio that
maximizes its cumulative risk-adjusted return by repeatedly rebalancing between
cash and a risky asset. Such problems can be solved using means of dynamic
programming or reinforcement learning.

