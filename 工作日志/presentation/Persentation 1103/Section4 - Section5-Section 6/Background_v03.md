---
marp: true
---
# Introduction

* This presentation introduces the industry context for machine learning in finance, discussing the critical events that have shaped the finance industry’s need for machine learning and the unique barriers to adoption.

* The finance industry has adopted machine learning to varying degrees of sophistication. How it has been adopted is heavily fragmented by the academic disciplines underpinning the applications.

---

* In particular, we begin to address many finance practitioner’s concerns that neural networks are a “black-box” by showing how they are related to existing well-established techniques such as linear regression, logistic regression, and auto regressive time series models.

* This presentation also introduces reinforcement learning for finance and is followed by more in-depth case studies highlighting the design concepts and practical challenges of applying machine learning in practice.
---
## 1 Background
* In 1955, John McCarthy, then a young Assistant Professor of Mathematics, at Dartmouth College in Hanover, New Hampshire, submitted a proposal with Marvin Minsky, Nathaniel Rochester, and Claude Shannon for the Dartmouth Summer Research Project on Artificial Intelligence (McCarthy et al. 1955). 
* These organizers were joined in the summer of 1956 by Trenchard More, Oliver Selfridge, Herbert Simon, Ray Solomonoff, among others. The stated goal was ambitious:“The study is to proceed on the basis of the conjecture that every aspect of learning or any other feature of intelligence can in principle be so precisely described that a machine can be made to simulate it. 

---
* An attempt will be made to find how to make machines use language, form abstractions and concepts, solve kinds of problems now reserved for humans, and improve themselves.” 
* Thus the field of artificial intelligence, or AI, was born.

---
* Since this time, AI has perpetually strived to outperform humans on various judgment tasks (Pinar Saygin et al. 2000). The most fundamental metric for this success is the Turing test—a test of a machine’s ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human (Turing 1995). 

---
* In recent years, a pattern of success in AI has emerged—one in which machines outperform in the presence of a large number of decision variables, usually with the best solution being found through evaluating an exponential number of candidates in a constrained high-dimensional space. 
* Deep learning models, in particular, have proven remarkably successful in a wide field of applications (DeepMind 2016; Kubota 2017; Esteva et al. 2017) including image processing (Simonyan and Zisserman 2014), learning in games (DeepMind 2017), neuroscience (Poggio 2016), energy conservation (DeepMind 2016), skin cancer diagnostics (Kubota 2017; Esteva et al. 2017).

---
* One popular account of this reasoning points to humans’ perceived inability to process large amounts of information and make decisions beyond a few key variables. 
* But this view, even if fractionally representative of the field, does no justice to AI or human learning. 
* Humans are not being replaced any time soon. The median estimate for human intelligence in terms of gigaflops is about 104 times more than the machine that ran alpha-go. Of course, this figure is caveated on the important question of whether the human mind is even a Turing machine.
---
### 1.1 Big Data — Big Compute in Finance
* The growth of machine-readable data to record and communicate activities throughout the financial system combined with persistent growth in computing power and storage capacity has significant implications for every corner of financial modeling.
* Since the financial crises of 2007–2008, regulatory supervisors have reoriented towards “data-driven” regulation, a prominent example of which is the collection and analysis of detailed contractual terms for the bank loan and trading book stresstesting programs in the USA and Europe, instigated by the crisis (Flood et al. 2016).
---
* “**Alternative data**”—which refers to data and information outside of the usual scope of securities pricing, company fundamentals, or macroeconomic indicators— is playing an increasingly important role for asset managers, traders, and decision makers. 

---
* Social media is now ranked as one of the top categories of alternative data currently used by hedge funds. Trading firms are hiring experts in machine learning with the ability to apply natural language processing (NLP) to financial news and other unstructured documents such as earnings announcement reports and SEC 10K reports. 
* Data vendors such as Bloomberg, Thomson Reuters, and RavenPack are providing processed news sentiment data tailored for systematic trading models.

---
* In de Prado (2019), some of the properties of these new, alternative datasets are explored: 
    * (a) many of these datasets are unstructured, non-numerical, and/or noncategorical, like news articles, voice recordings, or satellite images; 
    * (b) they tend to be high-dimensional (e.g., credit card transactions) and the number of variables may greatly exceed the number of observations; 
    * (c) such datasets are often sparse, containing NaNs (not-a-numbers); 
    * (d) they may implicitly contain information about networks of agents.
---
* Furthermore, de Prado (2019) explains why classical econometric methods fail on such datasets. 
* These methods are often based on linear algebra, which fail when the number of variables exceeds the number of observations. 
* Geometric objects, such as covariance matrices, fail to recognize the topological relationships that characterize networks. 
* On the other hand, machine learning techniques offer the numerical power and functional flexibility needed to identify complex patterns in a high-dimensional space offering a significant improvement over econometric methods.

---
* The “black-box” view of ML is dismissed in de Prado (2019) as a misconception. 

* Recent advances in ML make it applicable to the evaluation of plausibility of scientific theories; determination of the relative informational variables (usually referred to as features in ML) for explanatory and/or predictive purposes; causal inference; and visualization of large, high-dimensional, complex datasets.
---
* Advances in ML remedy the shortcomings of econometric methods in goal setting, outlier detection, feature extraction, regression, and classification when it comes to modern, complex alternative datasets.

---
* For example, in the presence of  $p$ features there may be up to $2^p-p-1$ multiplicative interaction effects. 
    * For two features there is only one such interaction effect $x_1x_2$.
    * For three features, there are $x_1x_2,x_1x_3,x_2x_3,x_1x_2x_3$ . 
    * For as few as ten features, there are 1,013 multiplicative interaction effects. 

---
* Unlike ML algorithms, econometric models do not “learn” the structure of the data. The model specification may easily miss some of the **interaction effects**. The consequences of missing an interaction effect, e.g. fitting $y_t=x_{1,t}+x_{2,t}+\epsilon_t$ instead of $y_t=x_{1,t}+x_{2,t}+x_{1,t}x_{2,t}+\epsilon_t$ , can be dramatic. 

---
* A machine learning algorithm, such as a decision tree, will recursively partition a dataset with complex patterns into subsets with simple patterns, which can then be fit independently with simple linear specifications. 
* Unlike the classical linear regression, this algorithm “learns” about the existence of the $x_{1,t}x_{2,t}$ effect, yielding much better out-of-sample results.
---
* There is a draw towards more empirically driven modeling in asset pricing research—using ever richer sets of firm characteristics and “factors” to describe and understand differences in expected returns across assets and model the dynamics of the aggregate market equity risk premium (Gu et al. 2018). 

---
* For example, Harvey et al. (2016) study 316 “factors,” which include firm characteristics and common factors, for describing stock return behavior. Measurement of an asset’s risk premium is fundamentally a problem of prediction—the risk premium is the conditional expectation of a future realized excess return.
* Methodologies that can reliably attribute excess returns to tradable anomalies are highly prized.

---
* Machine learning provides a non-linear empirical approach for modeling realized security returns from firm characteristics. 
* Dixon and Polson (2019) review the formulation of asset pricing models for measuring asset risk premia and cast neural networks in canonical asset pricing frameworks.

---
## 1.2 Fintech
* The rise of data and machine learning has led to a “fintech” industry, covering digital innovations and technology-enabled business model innovations in the financial sector (Philippon 2016). 
* Examples of innovations that are central to fintech today include cryptocurrencies and the blockchain, new digital advisory and trading systems, peer-to-peer lending, equity crowdfunding, and mobile payment systems. 

---
* Behavioral prediction is often a critical aspect of product design and risk management needed for consumer-facing business models;  consumers or economic agents are presented with well-defined choices but have unknown economic needs and limitations, and in many cases *do not behave in a strictly economically rational fashion*. 
* Therefore it is necessary to treat parts of the system as a *black-box* that operates under rules that cannot be known in advance.

---
### 1.2.1 Robo-Advisors
* Robo-advisors are financial advisors that provide financial advice or **portfolio management** services with minimal human intervention. 
* The focus has been on portfolio management rather than on estate and retirement planning, although there are exceptions, such as Blooom. 
* Some limit investors to the ETFs selected by the service, others are more flexible. 
* Examples include Betterment, Wealthfront, WiseBanyan, FutureAdvisor (working with Fidelity and TD Ameritrade), Blooom, Motif Investing, and Personal Capital. The degree of sophistication and the utilization of machine learning are on the rise among robo-advisors.
---
### 1.2.2  Fraud Detection
* In 2011 fraud cost the financial industry approximately $80 billion annually (Consumer Reports, June 2011). 
* According to PwC’s Global Economic Crime Survey 2016, 46% of respondents in the Financial Services industry reported being victims of economic crime in the last 24 months——a small increase from 45% reported in 2014. 16% of those that reported experiencing economic crime had suffered more than 100 incidents, with 6% suffering more than 1,000. 


---
* According to the survey, the top 5 types of economic crime are asset misappropriation (60%, down from 67% in 2014), cybercrime (49%, up from 39% in 2014), bribery and corruption (18%, down from 20% in 2014), money laundering (24%, as in 2014), and accounting fraud (18%, down from 21% in 2014). 

---
* Detecting economic crimes is one of the oldest successful applications of machine learning in the financial services industry. 

* See Gottlieb et al. (2006) for a straightforward overview of some of the classical methods: logistic regression, naïve Bayes, and support vector machines. 

* The rise of electronic trading has led to new kinds of financial fraud and market manipulation. Some exchanges are investigating the use of deep learning to counter spoofing.
---
### 1.2.3 Cryptocurrencies
* Blockchain technology, first implemented by Satoshi Nakamoto in 2009 as a core component of Bitcoin, is a distributed public ledger recording transactions. 
* Its usage allows secure peer-to-peer communication by linking blocks containing hash pointers to a previous block, a timestamp, and transaction data. 
* Bitcoin is a decentralized digital currency (cryptocurrency) which leverages the blockchain to store transactions in a distributed manner in order to mitigate against flaws in the financial industry.
* In contrast to existing financial networks, blockchain based cryptocurrencies expose the entire transaction graph to the public. 
* This openness allows, for example, the most significant agents to be immediately located (pseudonymously) on the network. 

---
* By processing all financial interactions, we can model the network with a high-fidelity graph, as illustrated in Fig 1 so that it is possible to characterize how the flow of information in the network evolves over time. 
![width:800px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104003040739.png)

---

![ width:1000px](https://xingqiu-tuchuang-1256524210.cos.ap-shanghai.myqcloud.com/2131/image-20221104003040739.png)
**Fig 1**$\qquad$*A transaction–address graph representation of the Bitcoin network. Addresses are represented by circles, transactions with rectangles, and edges indicate a transfer of coins. Blocks order transactions in time, whereas each transaction with its input and output nodes represents an immutable decision that is encoded as a subgraph on the Bitcoin network*.

---
* This novel data representation permits a new form of financial econometrics—with the emphasis on the topological network structures in the microstructure rather than solely the covariance of historical time series of prices. 
* The role of users, entities, and their interactions in formation and dynamics of cryptocurrency risk investment, financial predictive analytics and, more generally, in re-shaping the modern financial world is a novel area of research (Dyhrberg 2016; Gomber et al. 2017; Sovbetov 2018).
---

