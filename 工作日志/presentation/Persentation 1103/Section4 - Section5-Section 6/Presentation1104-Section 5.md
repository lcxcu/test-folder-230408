---
marp: true
---
# 5 Examples of Supervised Machine Learning in Practice

* The practice of machine learning in finance has grown somewhat commensurately
with both theoretical and computational developments in machine learning. Early
adopters have been the quantitative hedge funds, including Bridgewater Associates,Renaissance Technologies, WorldQuant, D.E. Shaw, and Two Sigma who have
embraced novel machine learning techniques although there are mixed degrees of
adoption and a healthy skepticism exists that machine learning is a panacea for
quantitative trading. In 2015, Bridgewater Associates announced a new artificial
intelligence unit, having hired people from IBM Watson with expertise in deep
learning. Anthony Ledford, chief scientist at MAN AHL: “It’s at an early stage. We
have set aside a pot of money for test trading. With deep learning, if all goes well,
it will go into test trading, as other machine learning approaches have.” Winton
Capital Management’s CEO David Harding: “People started saying, ‘There’s an
amazing new computing technique that’s going to blow away everything that’s gone
before.’ There was also a fashion for genetic algorithms. Well, I can tell you none
of those companies exist today—not a sausage of them.”

* Some qualifications are needed to more accurately assess the extent of adoption.
For instance, there is a false line of reasoning that ordinary least squares regression and logistic regression, as well as Bayesian methods, are machine learning
techniques. Only if the modeling approach is algorithmic, without positing a data
generation process, can the approach be correctly categorized as machine learning.
So regularized regression without use of parametric assumptions on the error
distribution is an example of machine learning. Unregularized regression with, say,
Gaussian error is not a machine learning technique. The functional form of the
input–output map is the same in both cases, which is why we emphasize that the
functional form of the map is not a sufficient condition for distinguishing ML from
statistical methods.

* With that caveat, we shall view some examples that not only illustrate some of
the important practical applications of machine learning prediction in algorithmic
trading, high-frequency market making, and mortgage modeling but also provide a
brief introduction to applications.

## 5.1 Algorithmic Trading

* Algorithmic trading is a natural playground for machine learning. The idea behind
algorithmic trading is that trading decisions should be based on data, not intuition.
Therefore, it should be viable to automate this decision-making process using
an algorithm, either specified or learned. The advantages of algorithmic trading
include complex market pattern recognition, reduced human produced error, ability
to test on historic data, etc. In recent times, as more and more information is being
digitized, the feasibility and capacity of algorithmic trading has been expanding
drastically. The number of hedge funds, for example, that apply machine learning for
algorithmic trading is steadily increasing.

* Here we provide a simple example of how machine learning techniques can
be used to improve traditional algorithmic trading methods, but also provide new
trading strategy suggestions. The example here is not intended to be the “best”
approach, but rather indicative of more out-of-the-box strategies that machine
learning facilitates, with the emphasis on minimizing out-of-sample error by pattern
matching through efficient compression across high-dimensional datasets.

* Momentum strategies are one of the most well-known algo-trading strategies;
In general, strategies that predict prices from historic price data are categorized
as momentum strategies.   Traditionally momentum strategies are based on certain
regression-based econometric models, such as ARIMA or VAR .
A drawback of these models is that they impose strong linearity which is not
consistently plausible for time series of prices.   Another caveat is that these models
are parametric and thus have strong bias which often causes underfitting.   Many
machine learning algorithms are both non-linear and semi/non-parametric, and
therefore prove complementary to existing econometric models.

* In this example we build a simple momentum portfolio strategy with a feedforward neural network. We focus on the S&P 500 stock universe, and assume we have
daily close prices for all stocks over a ten-year period

### Problem Formulation

* The most complex practical aspect of machine learning is how to choose the
input (“features”) and output. The type of desired output will determine whether
a regressor or classifier is needed, but the general rule is that it must be actionable
(i.e., tradable). Suppose our goal is to invest in an equally weighted, long only,
stock portfolio only if it beats the  $ S\&P 500$ index benchmark (which is a reasonable
objective for a portfolio manager). We can therefore label the portfolio at every
observation t based on the mean directional excess return of the portfolio:

$$
G_t=
 \begin{cases}
            1,  & \text{${1\over{N}}\sum_i {r^i_{t+h,t}-{\tilde{r}_{t+h,t}}}\geq \ \epsilon,$} \\
            0, & \text{${1\over{N}}\sum_i {r^i_{t+h,t}-{\tilde{r}_{t+h,t}}}< \ 0,$}\\
        \end{cases}
        \qquad \tag{18}
$$

* where $r^i_{t+h,t}$ is the return of stock i between times $t$ and $t + h$, $\tilde{r}_{t+h,t}$ is the return
of the $S\&P 500$ index in the same period, and $\epsilon$ is some target next period excess
portfolio return. Without loss of generality, we could invest in the universe $(N =
500)$, although this is likely to have adverse practical implications such as excessive
transaction costs. We could easily just have restricted the number of stocks to a
subset, such as the top decile of performing stocks in the last period. Framed this
way, the machine learner is thus informing us when our stock selection strategy
will outperform the market. It is largely agnostic to how the stocks are selected,
provided the procedure is systematic and based solely on the historic data provided
to the classifier. It is further worth noting that the map between the decision to hold
the customized portfolio has a non-linear relationship with the past returns of the
universe.


* To make the problem more concrete, let us set $h = 5$ days. The algorithmic
strategy here is therefore automating the decision to invest in the customized  portfolio or the $S\&P 500$ index every week based on the previous 5-day realized
returns of all stocks. To apply machine learning to this decision, the problem
translates into finding the weights in the network between past returns and the
decision to invest in the equally weighted portfolio. For avoidance of doubt, we
emphasize that the interpretation of the optimal weights differs substantially from
Markowitz’s mean–variance portfolios, which simply finds the portfolio weights to
optimize expected returns for a given risk tolerance. Here, we either invest equal
amounts in all stocks of the portfolio or invest the same amount in the $S\&P 500$
index and the weights in the network signify the relevance of past stock returns in
the expected excess portfolio return outperforming the market.

### Data
* Feature engineering is always important in building models and requires careful
consideration. Since the original price data does not meet several machine learning
requirements, such as stationarity and i.i.d. distributional properties, one needs to
engineer input features to prevent potential “garbage-in-garbage-out” phenomena.
In this example, we take a simple approach by using only the 5-day realized returns
of all $S\&P 500$ stocks. Returns are scale-free and no further standardization is
needed. So for each time $t$, the input features are
$$
X_t = \left[r^1_{t,t-5},\cdots,r^{500}_{t,t-5} \right] \qquad \tag(19)
$$

* Now we can aggregate the features and labels into a panel indexed by date. Each
column is an entry in Eq.19, except for the last column which is the assigned
label from Eq.18, based on the realized excess stock returns of the portfolio. An
example of the labeled input data $(X, G)$ is shown in Table 3.


**Table 3**  Training samples for a classification problem
![width:1000px](Presentation1104\fig8.png)

*  This example illustrates how algo-trading strategies can be
crafted around supervised machine learning.  Our model problem could be tailored for specific risk-reward and performance reporting metrics such as, for example,
Sharpe or information ratios meeting or exceeding a threshold.

* $\epsilon$ is typically chosen to be a small value so that the labels are not too imbalanced. As the value $\epsilon$ is increased, the problem becomes an “outlier prediction problem”—— a highly imbalanced classification problem which requires more advanced sampling and interpolation techniques beyond an off-the-shelf classifier.


* In the next example, we shall turn to another important aspect of machine
learning in algorithmic trading, namely execution. How the trades are placed is a
significant aspect of algorithmic trading strategy performance, not only to minimize
price impact of market taking strategies but also for market making. Here we shall
look to transactional data to perfect the execution, an engineering challenge by itself
just to process market feeds of tick-by-tick exchange transactions. The example
considers a market making application but could be adapted for price impact and
other execution considerations in algorithmic trading by moving to a reinforcement
learning framework.

* A common mistake is to assume that building a predictive model will result in a
profitable trading strategy. Clearly, the consideration given to reliably evaluating
machine learning in the context of trading strategy performance is a critical
component of its assessment.

## 5.2 High-Frequency Trade Execution

* Modern financial exchanges facilitate the electronic trading of instruments through
an instantaneous double auction. At each point in time, the market demand and
the supply can be represented by an electronic limit order book, a cross-section of orders to execute at various price levels away from the market price as illustrated in
Table 4.

![width:500px](Presentation1104\fig10.png)

**Table 4** This table shows a snapshot of the limit order book of $S\&P 500$ e-mini futures (ES).
The top half (“sell-side”) shows the ask volumes and prices and the lower half (“buy side”) shows
the bid volumes and prices. The quote levels are ranked by the most competitive at the center (the “inside market”), outward to the least competitive prices at the top and bottom of the limit order book. Note that only five bid or ask levels are shown in this example, but the actual book is much deeper

* Electronic market makers will quote on both sides of the market in an attempt to capture the bid–ask spread.  Sometimes a large market order, or a succession of smaller markets orders, will consume an entire price level.  This is why the market price fluctuates in liquid markets—an effect often referred to by practitioners as a “price-flip.”  A market maker can take a loss if only one side of the order is filled as a result of an adverse price movement.

* Figure.7 illustrates a typical mechanism resulting in an adverse price
movement. A snapshot of the limit order book at time $t$, before the arrival of a
market order, and after at time $t+1$ is shown in the left and right panels, respectively.
The resting orders placed by the market marker are denoted with the $“+”$ symbol — red denotes a bid and blue denotes an ask quote. A buy market order subsequently
arrives and matches the entire resting quantity of best ask quotes. Then at event time $t + 1$ the limit order book is updated—the market maker’s ask has been filled (blue minus symbol) and the bid now rests away from the inside market. The market marker may systematically be forced to cancel the bid and buy back at a higher price, thus taking a loss.

![width:800px](Presentation1104\fig11.png)

**Fig.7** A snapshot of the limit order book is taken at time t. Limit orders placed by the market marker are denoted with the “+” symbol—red denotes a bid and blue denotes an ask. A buy market order subsequently arrives and matches the entire resting quantity of best ask quotes. Then at event time t + 1 the limit order book is updated. The market maker’s ask has been filled (blue
minus symbol) and the bid rests away from the inside market. (Bottom) A pre-emptive strategy for avoiding adverse price selection is illustrated. The ask is requoted at a higher ask price. In this case, the bid is not replaced and the market maker may capture a tick more than the spread if both
orders are filled.


* Machine learning can be used to predict these price movements (Kearns and
Nevmyvaka 2013;   Kercheval and Zhang 2015;   Sirignano 2016;   Dixon et al. 2018;
Dixon 2018b,a) and thus to potentially avoid adverse selection.   Following Cont and
de Larrard (2013) we can treat queue sizes at each price level as input variables.
We can additionally include properties of market orders, albeit in a form which
our machines deem most relevant to predicting the direction of price movements
(a.k.a. feature engineering).   In contrast to stochastic modeling, we do not impose
conditional distributional assumptions on the independent variables (a.k.a. features)
nor assume that price movements are Markovian.

* We reiterate that the ability to accurately predict does not imply profitability
of the strategy.    Complex issues concerning queue position, exchange matching
rules, latency, position constraints, and price impact are central considerations for
practitioners.   Dixon (2018a) presents a framework for evaluating
the performance of supervised machine learning algorithms which accounts for
latency, position constraints, and queue position.    However, supervised learning is
ultimately not the best machine learning approach as it cannot capture the effect
of market impact and is too inflexible to incorporate more complex strategies. The  reinforcement learning could capture market impact and also flexibly formulate market making strategies.  

## 5.3 Mortgage Modeling

* Beyond the data rich environment of algorithmic trading, does machine learning
have a place in finance? One perspective is that there simply is not sufficient data
for some “low-frequency” application areas in finance, especially where traditional
models have failed catastrophically. The purpose of this section is to serve as a
sobering reminder that long-term forecasting goes far beyond merely selecting the
best choice of machine learning algorithm and why there is no substitute for strong
domain knowledge and an understanding of the limitations of data.

* In the USA, a mortgage is a loan collateralized by real-estate. Mortgages are
used to securitize financial instruments such as mortgage backed securities and
collateralized mortgage obligations. The analysis of such securities is complex and
has changed significantly over the last decade in response to the 2007–2008 financial
crises (Stein 2012).

* Unless otherwise specified, a mortgage will be taken to mean a “residential
mortgage,” which is a loan with payments due monthly that is collateralized by a
single family home. Commercial mortgages do exist, covering office towers, rental
apartment buildings, and industrial facilities, but they are different enough to be
considered separate classes of financial instruments. Borrowing money to buy a
house is one of the most common, and largest balance, loans that an individual
borrower is ever likely to commit to. Within the USA alone, mortgages comprise
a staggering $15 trillion dollars in debt. This is approximately the same balance as
the total federal debt outstanding (Fig.8).

![width:800px](Presentation1104\fig12.png)

**Fig.8**Total mortgage debt in the USA compared to total federal debt, millions of dollars,
unadjusted for inflation. Source: https://fred.stlouisfed.org/series/MDOAH,  https://fred.stlouisfed.org/series/GFDEBTN

* Within the USA, mortgages may be repaid (typically without penalty) at will by
the borrower. Usually, borrowers use this feature to refinance their loans in favorable
interest rate regimes, or to liquidate the loan when selling the underlying house. This has the effect of moving a great deal of financial risk off of individual borrowers,
and into the financial system. It also drives a lively and well developed industry
around modeling the behavior of these loans.

* The mortgage model description here will generally follow the comprehensive
work in Sirignano et al. (2016), with only a few minor deviations.
Any US style residential mortgage, in each month, can be in one of the several
possible states listed in Table 5.

![width:800px](Presentation1104\fig13.png)
**Table 5** At any time, the states of any US style residential mortgage is in one of the several possible states

* Consider this set of $K$ available states to be $\mathbb{K} = \{P ,C, 3, 6, 9,F,R,D\}$.
Following the problem formulation in Sirignano et al. (2016), we will refer to the
status of loan n at time $t$ as $U_t^n
n ∈ \mathbb{K}$, and this will be represented as a probability
vector using a standard one-hot encoding.

* If $X = (X1,...,XP)$ is the input matrix of $P$ explanatory variables, then we
define a probability transition density function $g$ : $\mathbb{R}^P → [0, 1]^{
K×K}$ parameterized
by $θ$ so that

$$
\mathbb{P}(U_{t+1}^n = i| U_t^n = j, X_t^n) = g_{i,j}(X_t^n|\theta),∀i,j \in \mathbb{K}\qquad \tag{20}
$$


* Note that $g(X_t^
n | θ )$ is a time in-homogeneous $K × K$ Markov transition matrix.
Also, not all transitions are even conceptually possible—there are non-commutative
states. For instance, a transition from $C$ to $6$ is not possible since a borrower cannot
miss two payments in a single month. Here we will write $p_{(i,j)} := g_{i,j} (X_t^
n | θ )$
for ease of notation and because of the non-commutative state transitions where$
p_{(i,j)} = 0$, the Markov matrix takes the form:
![width:800px](Presentation1104\fig14.png)

* Our classifier $g_{i,j} (X_t^
n | θ )$ can thus be constructed so that only the probability of transition between the commutative states are outputs and we can apply softmax functions on a subset of the outputs to ensure that $\sum_{j\in\mathbb{K} }g_{i,j}(Xt^n|\theta) = 1$ and hence the transition probabilities in each row sum to one.


* For the purposes of financial modeling, it is important to realize that both states
$P$ and $D$ are loan liquidation terminal states. However, state $P$ is considered to
be voluntary loan liquidation (e.g., prepayment due to refinance), whereas state
$D$ is considered to be involuntary liquidation (e.g., liquidation via foreclosure and
auction). These states are not distinguishable in the mortgage data itself, but rather the driving force behind liquidation must be inferred from the events leading up to the liquidation.

* One contributor to mortgage model misprediction in the run up to the 2008
financial crisis was that some (but not all) modeling groups considered loans
liquidating from deep delinquency $(e.g., status 9)$ to be the transition $9 → P$
if no losses were incurred. However, behaviorally, these were typically defaults
due to financial hardship, and they would have had losses in a more difficult
house price regime. They were really $9 → D $transitions that just happened to be
lossless due to strong house price gains over the life of the mortgage. Considering
them to be voluntary prepayments $(status P)$ resulted in systematic over-prediction
of prepayments in the aftermath of major house price drops. The matrix above
therefore explicitly excludes this possibility and forces delinquent loan liquidation
to be always considered involuntary.

* The reverse of this problem does not typically exist. In most states it is illegal
to force a borrower into liquidation until at least 2 payments have been missed.
Therefore, liquidation from $C$ or $3$ is always voluntary, and hence $C → P$
and $3 → P$. Except in cases of fraud or severe malfeasance, it is almost never
economically advantageous for a lender to force liquidation from $status 6$, but it is
not illegal. Therefore the transition $3 → D$ is typically a data error, but $6 → D$ is merely very rare.

#### Example 6     Parameterized Probability Transitions
If loan $n$ is current in time period $t$, then
$$
\mathbb{P}(U_t^n) = (0,1,0,0,0,0,0,0)^T \qquad\qquad \tag{21}
$$

If we have $p_{(c,p)} = 0.05, p_{(c,c)} = 0.9 $, and $p_{(c,3)} = 0.05$, then

$$
\mathbb{P}(U_{t+1}^n | X_t ^n) = g(X_t^n| \theta)\cdot\mathbb{P}(U_t^n) = (0.05,0.9,0.05,0,0,0,0,0)^T\qquad\qquad\tag{(22)}
$$

*  Common mortgage models sometimes use additional states, often ones that are
(without additional information) indistinguishable from the states listed above.
Table 6 describes a few of these

* The reason for including these is the same as the reason for breaking out states
like REO, status R. It is known on theoretical grounds that some model regressors
from $X_t^n$ should not be relevant for $R$. For instance, since the property is now owned
by the lender, and the loan itself no longer exists, the interest rate (and rate incentive)
of the original loan should no longer have a bearing on the outcome. To avoid
over-fitting due to highly colinear variables, these known-useless variables are then
excluded from transitions models starting in status $R$.

**Table 6** A brief description of mortgage states
![width:800px](Presentation1104\fig15.png)

* This is the same reason status $T$ is sometimes broken out, especially for logistic
regressions. Without an extra status listed in this way, strong rate disincentives
could drive prepayments in the model to (almost) zero, but we know that people
die and divorce in all rate regimes, so at least some minimal level of premature loan
liquidations must still occur based on demographic factors, not financial ones.

#### 5.3.1 Model Stability

* Unlike many other models, mortgage models are designed to accurately predict
events a decade or more in the future. Generally, this requires that they be built on
regressors that themselves can be accurately predicted, or at least hedged. Therefore,
it is common to see regressors like FICO at origination, loan age in months, rate
incentive, and loan-to-value (LTV) ratio. Often LTV would be called MTMLTV if
it is marked-to-market against projected or realized housing price moves. Of these
regressors, original FICO is static over the life of the loan, age is deterministic,rates can be hedged, and MTMLTV is rapidly driven down by loan amortization
and inflation thus eliminating the need to predict it accurately far into the future.

* Consider the Freddie Mac loan level dataset of 30 year fixed rate mortgages
originated through 2014. This includes each monthly observation from each loan
present in the dataset. Table 7 shows the loan count by year for this dataset.
When a model is fit on 1 million observations from loans originated in 2001


**Table 7** Loan originations by year (Freddie Mac, FRM30)

![Width:800px](Presentation1104\fig16.png)


When a model is fit on 1 million observations from loans originated in 2001 and observed until the end of 2006, its $C → P$ probability charted against age is shown in $Fig.9$.
![Width:800px](Presentation1104\fig17.png)
**Fig.9** Sample mortgage model predicting $C → 3$ and fit on loans originated in 2001 and observed until 2006, by loan age (in months). The prepayment probabilities are shown on the y-axis


* In $Fig.9$ the curve observed is the actual prepayment probability of the
observations with the given age in the test dataset, “Model” is the model prediction,and “Theoretical” is the response to age by a theoretical loan with all other
regressors from $X_t^n$ held constant. Two observations are worth noting:

1. The marginal response to age closely matches the model predictions; and
2. The model predictions match actual behavior almost perfectly



![Width:1000px](Presentation1104\fig18.png)
**Fig.10** Sample mortgage model predicting C → 3 and fit on loans originated in 2006 and observed until 2015, by loan age (in months). The prepayment probabilities are shown on the y-axis

* This is a regime where prepayment behavior is largely driven by age. When
that same model is run on observations from loans originated in 2006 (the peak of housing prices before the crisis), and observed until 2015, $Fig .10$ is produced.
Three observations are warranted from this figure:
1. The observed distribution is significantly different from $Fig.9$;
2. The model predicted a decline of 25%, but the actual decline was approximately
56%; and
3. Prepayment probabilities are largely indifferent to age.


* The regime shown here is clearly not driven by age. In order to provide even this level of accuracy, the model had to extrapolate far from any of the available data and “imagine” a regime where loan age is almost irrelevant to prepayment. This model meets with mixed success. This particular one was fit on only 8 regressors, a more complicated model might have done better, but the actual driver of this inaccuracy
was a general tightening of lending standards. Moreover, there was no good data series available before the crisis to represent lending standards.

* This model was reasonably accurate even though almost 15 years separated the start of the fitting data from the end of the projection period, and a lot happened in that time. Mortgage models in particular place a high premium on model stability, and the ability to provide as much accuracy as possible even though the underlying distribution may have changed dramatically from the one that generated the fitting
data. Notice also that cross-validation would not help here, as we cannot draw testing data from the distribution we care about, since that distribution comes from the future.

* Most importantly, this model shows that the low-dimensional projections of this (moderately) high-dimensional problem are extremely deceptive. No modeler would have chosen a shape like the model prediction from $Fig.9$ as function of age. That prediction arises due to the interaction of several variables, interactions that are not interpretable from one-dimensional plots such as this. As we will see in subsequent chapters, such complexities in data are well suited to machine learning, but not without a cost. That cost is understanding the “bias–variance tradeoff” and understanding machine learning with sufficient rigor for its decisions to be defensible.

