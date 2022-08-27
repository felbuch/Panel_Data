### Introduction

Lots of things are invisible, but we don't know how many, because we can't see them. I wish this quote were mine, but it's not. It's from Dennis, the menace:

![Denis](https://lh4.googleusercontent.com/p30ZyiHFccPtE1tyGqzBAyx7qWq3zYiwlS2L8MjcyK_gfWQfC37ys_2cwULZZFnotHzUaJQzTlRsg3m0TbJ_Sa4O4K1YGyUhKxZZRRjp0e3s9tczZkXoWQwp7AzT1JD3wOv4gMW2xHiaHmtE1A)

But it's true. And in the case of regression models, it's also a problem.

Suppose you're interested in studying the effect of company size on company profit. There are  lots of things that affect a company's profit but that can never be put into our regression model because we simply can't see them. For example, managerial ability, political connections, employee engagement or resistance to change. All these things help shape a company's profit. But we can't see them. We can't measure them. We can't put them into our model.

### What happens if you omit an important variable?

If we _do_ write a model to explain a company's profit in terms of its size, these variables will be missing. When an important variable is missing, this gives rise to an __omitted variable bias__. 

It is, indeed, a bias. Suppose $y$ depends on two variables, $x_1$ and $x_2$, so that $y$ can be written as $y=\beta_0+\beta_1x_1 +\beta_2x_2 +\epsilon$. However, $x_2$ is not in our model (because it's invisible). For simplicity, suppose $x_1$  is standardized and $\epsilon$ is independent of $x_1$.  Our estimate for $\beta_1$ is

$$
\hat\beta_1 = Cov(x_1,y) \\=Cov(x_1,\beta_0+\beta_1x_1+\beta_2x_2+\epsilon)=\\\beta_1 +\beta_2Cov(x_1,x_2)\\\therefore\\ \hat\beta_1\neq\beta_1
$$

In plain English, this means our estimates of how company size relates to company profits will be _wrong_.  It will be biased, and the size of this bias is $\beta_2 Cov(x_1,x_2)$. This is proportional to:

* the effect that the omitted variable has on $y$ (i.e., how much this omitted variable really matters), $\beta_2$;
* how much $x_2$ is related to $x_1$, $Cov(x_1,x_2)$. If they are related, the effect of $x_2$ will affect $y$ _through_ $x_1$. You can think of the effect of $x_2$ creeping into $y$ by $x_1$, as some sort of alien parasite, like Venom. For $x_2$ to "reach" $y$ through $x_1$, however, $x_2$ must first have access to $x_1$. The measure of how much "access" $x_2$ has to $x_1$ is given by $Cov(x_1,x_2)$.

In short, if we omit a variable ($x_2$) that matters ($\beta_2\neq0$) and can make its importance felt ($Cov(x_1,x_2)\neq0$), our estimates of the model's coefficients will be wrong ($\hat\beta_1\neq\beta$). Managerial ability is a relevant variable to explain profits, and bigger companies have access to more qualified managers, thus satisfying  both these conditions. As a result, a regression of company profits on company size will give us a wrong estimate of how these two variables are related. 

So if we _are_ interested in studying the effect of company size on company profits, what can we do? 

### Panel Data

One thing we can do is observe our companies not just once, but throughout time. Intuitively, this allows us to "get to know" our companies, and know "how they usually perform". This is a way of indirectly learning the overall effect of those variables we cannot directly observe.

__Panel data__ is just that. We select a group of companies and then, periodically (say, every year), collect the same (updated) data for those companies. This way, we can see how the sizes of these companies are evolving, and how their profits is responding. Note how this is different from __cross-sectional data__, where all of our companies appear once, and only once, in our dataset. We don't see them evolving through time.  Thus, while an observation in cross-sectional data is represented as $(x_i,y_i)$, $i=1,...,N$, in panel data, an observation is represented as 
$$
(x_{it}, y_{it}) \\i=1,...N \\ t=1,...,T
$$
So, for example, $Size_{it}$ is the size of company $i$ at year $t$ . Here, $N$ companies are being observed during $T$ years.

### The Panel Data fundamental equation

With this notation, we can express our problem more clearly. To make things as simple as possible, consider a single observed variable and a single unobserved variable.

Our variable of interest is given by an expression of the form
$$
Y_{it} = \beta_0 + \beta_1 X_{it}+U_i+\epsilon_{it}
$$
where $X_{it}$ is the observed variable, $U_i$ is an unobserved variable and $\epsilon_{it}$ is the idiosyncratic error. Our goal is to estimate $\beta_1$.

Note that __$Y_{it}$ depends on both observed and unobserved variables__. It's also important to note that the expression above is the _true_ expression for $Y_{it}$. It's not the model we actually write down, because we cannot observe $U_i$. In our dataset, we have the values for the Y's and for the X's, but we don't have the values for the U's.

Also note that we write $U_i$, but not $U_{it}$. There's no "t". This means we consider these unobserved variables to be an immutable characteristic of the individual. In other words, it's a characteristic that each individual has, has always had, and will always have (at least throughout the duration of our study). This makes things much easier, and we can always change this later. So bear with me.

Since $U_i$ is an unobserved characteristic from the individual it's also called __unobserved heterogeneity__. It also has other names, but I like that name very much. "Hetero" means "different", and "genesis" means "creation", or "origin". So "heterogeneity" means "the origin of differences". Indeed $U_i$ is that essence that makes each individual unique, and unlike any other individual. Even if two individuals _look_ exactly equal (i.e., they have the same _observed variables_, the same $X_{it}$), they will still _be_ different because of their different values of $U_i$.

### Panel data techniques

(Almost) all panel data techniques are different answers to the same problem: how to estimate $\beta_1$  without knowing the value of $U_i$. Somehow, these techniques will try to eliminate $U_i$ without changing the value of $\beta_1$.

There are different ways to do this, and hence different panel data techniques:

* Pooled regression
* Random Effects
* Fixed Effects
* First Difference models

Let's take a look at each of these.

### Pooled Regression

Pooled regression solves a problem by ignoring it (so like me!) 

It just runs a linear regression of $Y_{it}$ on $X_{it}$ ignoring altogether the existence of $U_i$.

We've already shown this gives rise to omitted variable bias. But there is one case where this bias doesn't happen: if the omitted variable is uncorrelated with $X_{it}$, then they have no way of affecting $Y_{it}$, and our estimates of $\beta_1$ will not be biased.

So in this particular case where $X_{it}$ and $U_i$ are independent, a good, old classical regression works just fine.  The big question is: how do we know if the $X_{it}$ and the $U_i$ are independent, since we don't observe $U_i$? That's the question answered by the Breusch-Pagan Lagrange Multipliers test.

#### The Breusch-Pagan Lagrange Multipliers test

The Breusch-Pagan Lagrange Multiplier test (BPLM) tests whether it is safe to perform a classical linear regression or whether we actually need to take the $U_i$s into account. It can be thought of as a test to choose between two models:
$$
H_0: \text{Pooled Regression} \\
H_a: \text{Random Effects}
$$
If the null hypothesis is rejected, this means we cannot neglect the existence of the $U_i$s and we're directed towards another kind of model, called a Random Effects model. Then, other statistical tests can be used to see if a Random Effects model will suffice or whether we'll need yet another kind of model.

For now, let's give some intuition behind the BPLM test. 

By omitting the $U_i$s, Pooled Regression is effectively incorporating the $U_i$s into the errors of our model. So our Pooled Regression model can be though of as
$$
Y_{it} = \beta_0 + \beta_1 X_{it}+\nu_{it} \\
\nu_{it}=U_i+\epsilon_{it}
$$
In classical regression, residuals must be independent. If the $U_i$s do not exist (or if they're all equal), then this is fine. But if the $U_i$s exist and are different from one another, each $U_i$ will affect all the residuals corresponding to that $i$. If $U_i$ is large, all residuals of that $i$ will also likely be large, and vice-versa. Hence, if $U_i$s exist and are different from one another, residuals will not be independent. Indeed, if we calculate the covariance of two residuals, we get:
$$
\nu_{it}=U_i+\epsilon_{it}\implies Cov(\nu_{it},\nu_{is})=\sigma_U^2
$$
The Breusch-Pagan test tests if residuals are independent in order to verify that $\sigma_U^2=0$. If this is the case, all $U_i$s are equal, and can therefore be ignored.

If not, we need a more sophisticated model. Random Effects model is next in line. It acknowledges the existence of different $U_i$s for different individuals, but it still requires them to be independent from the $X_{it}$s.  

### Random Effects

Of all models we'll see, Random Effects is the first model that acknowledges the existence of $U_i$ and tries to deal with it, rather than just ignore it. It still makes a very strong assumption, however:

> The $U_i$s are uncorrelated with the $X_{it}$s

Or, to put it mathematically, $Cov(U_i,X_{it})=0$.

Random Effects model acknowledge that, since $U_i$ cannot be placed in our model, it will be thrown to the residuals. Our model can therefore be though of as:
$$
Y_{it} = \beta_0 + \beta_1 X_{it}+\nu_{it} \\
\nu_{it}=U_i+\epsilon_{it}
$$
where $\nu_{it}$ is called a __composite error__, because it is composed of both $U_i$ and $\epsilon_{it}$.

One problem of running a pooled regression in this case is that the composite errors will be serially correlated. As we've shown before, $Cov(\nu_{it},\nu_{is})=\sigma_U^2$ for $t\neq s$. Ordinary Least Squares cannot deal with autocorrelation between errors, so we need a new estimation method: Generalized Least Squares.

#### Generalized Least Squares

In vanilla Least Squares regression, we make the assumption that all covariances are zero and all variances are equal. This means the covariance matrix of the residuals is proportional to the identity matrix: $\sigma^2\mathbf{I}$. In Generalized Least Squares (GLS), we allow this matrix to be any (positive-definite) matrix, $\mathbf{\Omega}$. 

Our objective function also changes. Rather than seeking to minimize the sum of squares, $(\mathbf{y}-\mathbf{X\hat{\beta}})^T(\mathbf{y}-\mathbf{X\hat{\beta}})$, we now seek to minimize $$(\mathbf{y}-\mathbf{X\hat{\beta}})^T\mathbf{\Omega}^{-1}(\mathbf{y}-\mathbf{X\hat{\beta}})$$ . 

To gain some intuition on what this implies, forget about covariances for just a second. In this case, $\mathbf{\Omega}$ is being representative of residual's variances, and thus multiplying by $\mathbf{\Omega}^{-1}$  is symbolically similar to dividing by the residuals' variance. Thus, both $(\mathbf{y}-\mathbf{X\hat{\beta}})^T$ and $(\mathbf{y}-\mathbf{X\hat{\beta}})$ get divided by the standard deviation. In practice, this means we're optimizing the sum of squares of _standardized_ residuals, rather than the sum of squares of residuals, plain and simple. If we bring back covariances into the picture, then our objective function can be thought of as a _Mahalanobis_ distance (rather than a Euclidian distance) between $\mathbf{y}$ and $\mathbf{X\hat\beta}$.

The value of $\hat\beta$ can be obtained from Calculus. We get:
$$
\hat\beta = \mathbf{(X^T\Omega^{-1}X)^{-1}X^T\Omega^{-1}y}
$$
Note that setting $\mathbf{\Omega}=\sigma^2\mathbf{I}$, as in the classical case, yields the classical estimate for $\hat{\beta}_{OLS}=(X^TX)^{-1}X^Ty$.

Random Effects can be thought of as a GLS model where  $\mathbf{\Omega}$ has a special structure, reflecting the variances and covariances of residuals of the form $\nu_{it}=U_i+\epsilon_{it}$. The diagonal elements of $\mathbf{\Omega}$ are $\mathbb V({\nu_{it}})=\sigma_U^2+\sigma_\epsilon^2$, which we assume to be constant.  As for its off-diagonal terms, they're $Cov(\nu_{it},\nu_{is})$, which we've already shown to be equal to $\sigma_U^2$. Hence, in the context of Random Effects,
$$
\mathbf{\Omega}=\left(\begin{array}{ccccc}
\sigma_U^2+\sigma_\epsilon^2 & \sigma_U^2 & \sigma_U^2 & ... & \sigma_U^2\\
\sigma_U^2 & \sigma_U^2+\sigma_\epsilon^2 & \sigma_U^2 & ... & \sigma_U^2\\
\sigma_U^2 & \sigma_U^2 & \sigma_U^2+\sigma_\epsilon^2 & ... & \sigma_U^2\\
... & ... & ... & ... & ...  \\
\sigma_U^2 & \sigma_U^2 & \sigma_U^2 & ... & \sigma_U^2+\sigma_\epsilon^2\\
\end{array}\right)
$$
  Writing the matrix this way evidences a problem with GLS: we do we know the value of $\sigma_U^2$, if $U$ is unobserved? And how do we know it _before_ running the model, given that we need $\mathbf{\Omega}$ in order to calculate $\beta$? 

The answer is we don't. As it is conceived, GLS is not feasible. But we can tweak it to make it feasible. We can proceed on the following steps:

1. Start with an initial estimate of $\mathbf{\Omega}$ , say, $\mathbf{\hat\Omega_0}=\mathbf{I}$.
2. Use the variances and covariances of the estimates residuals to build a new estimate of $\mathbf{\Omega}$,  which we'll call $\mathbf{\hat\Omega_1}$. 
3. Repeat until convergence i.e. until $\mathbf{\hat\Omega_{n+1}}\approx \mathbf{\hat\Omega_{n}} $ 

This procedure is called __Feasible GLS__ because it's makes GLS be feasible.

#### The Hausman test

Random Effects makes a very strong assumption: that the unobserved variable, $U_{i}$ be uncorrelated with the observed variable, $X_{it}$. This may not be realistic. It may be easier for bigger companies to hire better managers, for instance, thereby imposing a correlation between size and managerial ability. So how do we know if it makes sense to run a Random Effects model?

The Hausman test answers precisely this question. Just like the Breusch-Pagan test, it can be thought of as a test between two kinds of models:
$$
H_0: \text{Random Effects} \\
H_a: \text{Fixed Effects}
$$
In this respect, the Hausman test works as a next step after the Breusch-Pagan test. Once the Breusch-Pagan test rejects a Pooled Regression in favor of a Random Effects, the Hausman test whether we should stick with a Random Effects model or moved towards a Fixed Effects approach.

### Fixed Effects

Whereas Pooled Regression and Random Effects, through the unobserved variable into the residual term, fixed effects does something totally different: it tries to eliminate it.

Intuitively, it does so by comparing each individual to its own average. Recall our example of relating company size to company profits. Some companies may have higher profits because their managers are better. But we can compare each company to its own average profits. Firms with higher managerial ability will have higher profits on average, so by comparing each firm's profits with its own average profits, we can control for managerial ability even without directly seeing it.

To see how this works mathematically,  recall our fundamental equation:
$$
Y_{it} = \beta_0 + \beta_1 X_{it}+U_i+\epsilon_{it}
$$
The average of $Y$ of company $i$ is
$$
\bar{Y}_i=\beta_0+\beta_1\bar{X}_i+U_i
$$
Note that, since we assume $U_i$ is _fixed in time_, it's historical average is simply $U_i$ itself. So if we subtract one equation from the other, the $U_i$s cancel each other out and we get:
$$
(Y_{it}-\bar{Y}_i)=\beta_1(X_{it}-\bar{X}_{it})+\epsilon_{it}
$$
This equation is called a __reduced equation__, and note that it does not contain any unobserved variable. Rather than regression $Y_{it}$ on $X_{it}$ directly, we can simply regress $(Y_{it}-\bar{Y}_i)$ on $(X_{it}-\bar{X}_i)$. This is what Fixed Effects does. The process of subtracting a variable's historical average from itself is called __demeaning__. It's also called a __within transformation__.

Why is this model called _fixed_ effects? Because it exploits the fact that the $U_i$s are _fixed_ to the individual $i$, not varying from one time period to another. Thus, the $U_i$s are called __individual fixed effects__. 

We can also have __time fixed effects__, which would be unobserved variables that affect all companies on a given year, but not on other years, such as a financial crisis, for example. We can also have both individual and time fixed effects.

Fixed effects are thus quite flexible. Also, note that they make no assumptions whatsoever regarding the correlation between $U_i$ and $X_{it}$, so dynamics such as bigger companies having access to better managers do not compromise our model.

One issue with fixed effects, though, is that __variables that do not change with time get eliminated__ from the model. This can be a problem if we're interested in variables such as CEO gender, or country of the company's headquarters.

 ### First Difference models

Perhaps the easiest way to see the impact of one variable on another is to see how a change in one variable relates to a change in the other variable. This is what First Difference models do. Rather than regression $Y_{it}$ on $X_{it}$, they regress a _change_ in $Y_{it}$ on a _change_ in $X_{it}$, i.e., $\Delta Y_{it}$ on $\Delta X_{it}$.

It turns out that doing this eliminates the $U_i$s, similarly to what happens in the Fixed Effects model. To see this in action, let's start from our fundamental equation:
$$
Y_{it} = \beta_0 + \beta_1 X_{it}+U_i+\epsilon_{it}
$$
Now let's look at the same equation for the previous period:
$$
Y_{it-1} = \beta_0 + \beta_1 X_{it-1}+U_i+\epsilon_{it-1}
$$
Since $U_i$ doesn't change from one time period to the next, it gets canceled out when we subtract one equation from the other:
$$
(Y_{it}-Y_{it-1}) = \beta_1 (X_{it}-X_{it-1})+(\epsilon_{it}-\epsilon_{it-1})
$$
or, using the deltas notation, $\Delta Y_{it} = \beta_1 \Delta X_{it} + \Delta \epsilon_{it}$

 Just as in Fixed Effects, we have succeeded in eliminating $U_i$. And, just as in Fixed Effects, variables that do not change in time, such as CEO gender, get eliminated. What's _different_ from the Fixed Effects model, though, is that here we are not comparing each company's profit with it's historical average profit, but with it's profit on the previous year. But there's another important difference between the Fixed Effects and the First Difference models: the error term. 

In Fixed Effects, subtracting the mean of the error term from itself does not change it, because the mean error is zero. So the error term in the reduced equation is the same error that figures in the original equation for $Y_{it}$. In a First Difference model, however, the error gets subtracted from the error in the previous period. Our error at time $t$ is thus $\epsilon_{it}-\epsilon_{it-1}$, and our error at time $t-1$ is $\epsilon_{it-1}-\epsilon_{it-2}$. Note that both terms contain $\epsilon_{it-1}$. Are these two errors independent?

The answer is no. Indeed, if we calculate the covariance between them, we can see that it's not zero:
$$
Cov(\epsilon_{it}-\epsilon_{it-1}, \epsilon_{it-1}-\epsilon_{it-2})=\\Cov(\epsilon_{it},\epsilon_{it-1})+Cov(\epsilon_{it},\epsilon_{it-2})+Cov(\epsilon_{it-1},\epsilon_{it-1})+Cov(\epsilon_{it-1},\epsilon_{it-2}) =\\0+0+\sigma^2+0=\sigma^2\neq 0
$$
Classical regression models require the errors to be independent, and First Differences is no exception. This means that in First Differences model, the $\Delta \epsilon_{it}$s must themselves obey
$$
\Delta\epsilon_{it}\sim N(0;\sigma^2)
$$
In other words,
$$
\Delta\epsilon_{it} = \eta_{it} \\ \eta_{it}\sim N(0;\sigma^2)
$$
But this is equivalent to saying that
$$
\epsilon_{it} = \epsilon_{it-1} + \eta_{it} \\
\eta_{it} \sim N(0;\sigma^2)
$$
Thus, in order for the errors in the First Difference model to be independent, the original errors must follow a very specific process: each error must be the previous error added to some random noise. The technical way of saying this is saying that the error terms in the original equation follow a __random walk__.

This is important, because it helps us know when we should do a Fixed Effects model and when we should do a First Difference model.

#### Choosing between Fixed Effects and First Differences

Let's say you run a Fixed Effects. If residuals are serially uncorrelated, this means errors in the original equation were serially uncorrelated. Running a First Differences approach is not a good idea, because residuals from this model will _not_ be serially uncorrelated, as we have shown. So if you run a Fixed Effects an residuals are serially uncorrelated, stick with your Fixed Effects.

If, however, residuals follow a random walk, then a First Difference model is the way to go, since (as we have also shown) residuals in a First Difference model will be uncorrelated if the original residuals follow a random walk.

But how do we know if residuals follow a random walk?

One way is by performing a __Breusch-Godfrey test__ (not to be confused with Breusch-Pagan) on the $\Delta \epsilon_{it}$s and _failing_ to reject the null hypothesis. Another is by using the __Durbin-Whatson statistic__.

The Breusch-Godfrey test consists in regressing $\Delta\epsilon_{it}$ on $\Delta \epsilon_{it-1}$, controlling for $\Delta X_{it}$ and testing the null hypothesis that all coefficients are zero. If this hypothesis is not rejected, then $\Delta \epsilon_{it}$ is just a random error term with no serial correlation. First differences work.

The Durbin Whatson statistic is another approach to the same problem. This statistic, applied to $\Delta \epsilon_{it}$, can be written
$$
DW={\sum_{t=0}^T (\Delta\epsilon_{it}-\Delta\epsilon_{it-1})^2\over \sum_{t=0}^T (\Delta\epsilon_{it})^2}
$$
If $\Delta \epsilon_{it}$ follows a random walk, the difference in the numerator should be zero except for a random term. One can prove that, in this case, the Durbin-Whatson statistic should be close to $2$. Indeed, one can also show that the Durbin-Whatson statistic is approximately
$$
DW\approx2\cdot(1-\rho_{\Delta\epsilon_{it},\Delta\epsilon_{it-1}})
$$
where $\rho_{\Delta\epsilon_{it},\Delta\epsilon_{it-1}}$ is the correlation between successive residuals of the First Differences model. If these residuals are independent, then the correlation between them should be zero and the Durbin-Whatson statistic will be roughly 2. There are tabulated values for critical values of the DW statistic, so one can use this statistic to perform a hypothesis testing. As a rule of thumb, however, $DW < 1$ is often taken as a warning sign of positive serial correlation between residuals, and thus as a warning not to use First Differences.

So if the Fixed Effects residuals are independent, we go with the fixed effects. If not, we can test if they follow a random walk by applying a Breusch-Godfrey test or using the Durbin-Whatson statistic on the residuals for the First Difference model. But what if neither of these two things happen? What if the Fixed Effects residuals are _not_ independent but also _don't_ follow a random walk?

### Corrections for residuals with serial correlation following an AR(1)

> What if the Fixed Effects residuals are _not_ independent but also _don't_ follow a random walk?

Our previous question is a bit too broad. Let's make it a bit more specific?

> What if residuals follow an AR(1) process?

What is an AR(1) process? 

#### Autoregressive processes of the first order

An Autoregressive Process of the first order - or AR(1), for short - is a process where each term affect the next term. So if a result was above average today, chances are it will be above average tomorrow. More technically, error terms follow an AR(1) process if
$$
\epsilon_{t} = \rho\epsilon_{t-1}+\eta_{t} \\
\text{where} \\|\rho|<1
$$
In most real-world situations, $\rho$ is positive, meaning that if something was above average on one year, it will likely be above average the next year. If a company is growing, for example, and it grew considerably in a certain year, it is likely that it will also grow the next year. Positive $\rho$ means _inertia_ and, just like inertia, it's all around us.

Negative $\rho$ also exist. It means that a result above average in one year tends to be followed by a result below average the next year. This is the case for things that exhibit a reversal-to-the-mean behavior, such as commodities prices and the (discretized) Ornstein–Uhlenbeck process. 

But while negative $\rho$ exists, positive $\rho$ is much more common.

There are two borderline cases of AR(1) processes that will be of special interest to our subject. The first is what happens when $\rho=0$. In this case, there's no dependence between errors. We have __serially uncorrelated residuals__. This is, to be fair, a degenerate case of AR(1), because there is no autoregressive behavior whatsoever. But it's useful to know that, at one end of the $|\rho|$ spectrum, residuals are independent. What happens on the other end of the spectrum?

The other end of the spectrum is $\rho=\pm 1$. Let's think positive and focus on $\rho=+1$. In this case, $\epsilon_t = \epsilon_{t-1} + \eta_t$, meaning that errors _are_ serially correlated in a very special way: they follow a __random walk__.

So at $\rho=0$, we're in the world where Fixed Effects work best, and at $\rho=1$ we're in the world where First Differences is at its prime. What happens when $\rho$ lies _between_ these two extremes?

#### Dealing with AR(1)-serially correlated residuals in Panel Data

If $\rho$ lies between zero and 1, residuals are not independent. However, since $\epsilon_{t}=\rho\epsilon_{t-1}+\nu_{t}$, it follows that $\epsilon_{t}-\rho\epsilon_{t-1}$ _are_ independent. This gives us a hint on how to transform a panel data regression with correlated residuals into another regression whose residuals are independent. Begin by our fundamental equation:
$$
Y_{it} = \beta_0 + \beta_1 X_{it}+U_i+\epsilon_{it}
$$
Now, let's write the expression for $\rho Y_{it-1}$:
$$
\rho Y_{it-1} = \rho\beta_0 + \rho\beta_1 \rho X_{it-1}+\rho U_i+\rho \epsilon_{it-1}
$$
Subtracting one equation from the other yields
$$
(Y_{it}-\rho Y_{it-1})=(1-\rho)\beta_0 + \beta_1(X_{it}-\rho X_{it-1})+(1-\rho)U_i + (\epsilon_{it}-\rho\epsilon_{it-1})
$$
Note that in this last equation, the error terms are equal to $\nu_{it}$, and hence are _serially uncorrelated_. The problem of serially correlated residuals has been solved. 

If we had subtracted one equation from the other, i.e., $Y_{it-1}$ from $Y_{it}$, this process would have been called differentiation. But we didn't do exactly that. We subtracted only a fraction $\rho$ of  $Y_{it-1}$ from $Y_{it}$. For this reason, this process is called __quasi-differentiation__. Speakers of romance languages will recognized the oddly written "quasi" as a synonym for "almost". Indeed, when $\rho=1$, we have 100% differentiation, and the expression above becomes equivalent to a First Difference model's equation.

Quasi-differentiation solves the problem of autocorrelated residuals. It's useful to note, however, that it isn't a silver bullet for all types of serial correlation problems. It only works when residuals are correlated as an AR(1) process. 

Another issue of quasi-differentiation is that it requires us to know the value of $\rho$, which is seldom the case in practice. One way out of this conundrum is this:

1. Estimate the regression assuming $\rho=0$. Get the residuals from this equation.
2. Regress $\epsilon_{it}$ on $\epsilon_{it-1}$ to get a new estimate of $\rho$, say $\hat\rho_1$
3. Estimate the quasi-differentiated regression assuming this new value of $\rho$
4. Repeat until convergence i.e. until $\hat \rho_{n+1} \approx \hat \rho_n$

This is the rationale behind the __Cochrane-Orcutt__ and the __Prais-Winsten__ estimation methods. They have a minor difference, and the Prais-Winsten is technically a bit superior, because it makes use of the first observation, which the Cochrane-Orcutt method discards. A slightly different approach is the __Hildreth-Lu__ method. It tries many different values for $\rho$ (spaced over the [0;1] interval) and picks the one with the best fit. Whereas Cochrane-Orcutt and Prais-Winsten use an _iterative_ approach, Hildreth-Lu uses a _grid-search_ through a one-dimensional parameter space. 

One final point that should be mentioned is that quasi-differentiation doesn't completely eliminate the effect of the unobserved variable, $U_i$. Unlike in a First Differences model or in a Fixed Effects model, where the $U_i$ get's cancelled out, in quasi-differentiation the $U_i$ is sill there. It did, however, get weaker, as it's getting multiplied by $1-\rho$, which is a number between 0 and 1. So quasi-differentiation weakens the effect of the unobserved variable, but doesn't completely eliminate it. It's a good thing, but it may not be enough.

So maybe we should try something totally different....?

### Instrumental Variables

The problem of panel data is ultimately a problem of variable omission. Since $U_i$ is invisible, it cannot be put directly into the regression model, so it is omitted. But this omission has consequences: namely, the coefficient of the visible variable $X_{it}$ is biased by $Cov(X_{it},U_i)$ . What would happened if we found a variable that could substitute for $X$ _without_ being correlated to $U_i$? In this case, the bias would be zero, so omitting $U_i$ wouldn't be a big deal. This is the rationale behind the method of instrumental variables.


> An __instrumental variable__ of $X$ is another variable, $Z$, that affects $Y$ _only through_ X. The technical requirements for this is that $Z$ be correlated to $X$ but not to the omitted variable $U$.

A classical example is trying to predict the impact of cigarette usage on health. There are lots of things that can impact health besides cigarettes, so there clearly are omitted variables in this regression. One instrument we can use is cigarette tax. While it is correlated to cigarette usage, it is not correlated to any other variable impacting health.

One thing to bear in mind is that, while we can test the hypothesis that $Z$ and $X$ are correlated, we _cannot_ test the hypothesis that $Z$ and $U$ are uncorrelated. This is because $U$ is, by definition, unobservable. This means the choice of an instrument always relies on non-verifiable arguments. One can argue that an instrument variable is not correlated to other unobserved variables based on logical reasoning, on plausibility, and on substantive knowledge from Economics, Sociology, or whatever the field may be. But these are all arguments -- they're not statistical tests, and they're not grounded on data.

This is one of the main weaknesses in instrumental variables.

Another issue with instrumental variables is the use of so-called __weak instruments__. An instrumental variable $Z$ is a weak instrument if its correlation with $X$ is weak. In this case, the instrumental variable method performs poorly and, in some cases, can even perform worse than a classical linear regression where $U_i$ is simply ignored and nothing is done about it. This is because the bias of the estimated coefficient depends on the ratio ${Cov(Z,U) \over Cov(Z,X)}$, so if $Z$ and $X$ are weakly correlated ($Cov(Z,X)\approx0$), the bias can get large even if $Cov(Z,U)$ is small. This is known as the _weak instrument problem_.

Finally, proper use of the instrumental variables method requires the relationship between $X$ and $Y$ be __structural__. This means that $X$ must _cause_ $Y$, not merely be correlated to it. Proving causality is a challenge on its own sake and, of course, one may ultimately find that $X$ does not cause $Y$. But that's not the point. The point is that, at least in theory, $X$ causes $Y$, so when we introduce  the instrumental variable $Z$, we're using correlation as an instrument to probe causation. The underlying premise is that, though correlation does not imply causation, we do expect to find correlation if causation is present. Unfortunately, however, this isn't true: not only does correlation not imply causation, but causation also does not imply correlation. Still, it often does. And that's the assumption made by the instrumental variable method: if we find correlation in a causal relationship, this correlation is evidence of the existence of the causation.  

How are regression coefficients estimated in the presence of instrumental variables? There are different ways to answer this question, but a very popular one is to think of a regression with instrumental variables as a __two-stage least squares (2SLS)__ . In the first stage, regress $X$ on $Z$ and save the predicted values of $X$, which we'll call $\hat X$. Then, on the second stage, regress $Y$ not on $X$, but on $\hat X$.

We could, of course, do the maths this time. But it gets messy, adds little intuition and looks just like a least squares problem, where you input the result of the first step into the second step. 

What's the biggest challenge in the instrumental variables approach? Finding a good instrument.

Yet, for one kind of problems, a useful instrument is readily available. Let's talk about this kind of problems next...

### Dynamic panels

A dynamic panel is a panel where each time period is affected by previous time periods. In other words, in a dynamic panel, $Y_{it}$ depends on some past value of $Y$, such as $Y_{it-1}$. Our fundamental equation looks like this:
$$
Y_{it} = \beta_0 + \beta_1 X_{it}+\beta_2 Y_{it-1}+ U_i+\epsilon_{it}
$$
Since $Y_{it-1}$ depends on $\epsilon_{it-1}$, this means $Y_{it}$ also depends on $\epsilon_{it-1}$. But since $Y_{it-1}$ depends on $Y_{it-2}$, then $Y_{it}$ also depends on $\epsilon_{it-2}$. This goes on _ad infinitum_, and you get that all data points get intertwined, each one depending on the previous ones. This is a problem for classical linear regression, where data points are supposed to be independent.

Fortunately, there is a way to solve this problem. Since $\epsilon_{it-1}$ affects $Y_{it}$ _only_ through $Y_{it-1}$, we can use it as an instrumental variable. That's the rationale behind the __Arellano-Bond estimator__ (except for the technical detail that we actually use $Y_{it-1}$, not $\epsilon_{it-1}$ as the instrument)





