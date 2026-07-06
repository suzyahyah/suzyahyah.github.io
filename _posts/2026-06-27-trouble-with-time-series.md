---
layout: post
title: "The Unreasonable Difficulty of Time Series Forecasting"
date: 2026-06-27
mathjax: true
status: [Notes]
categories: [Machine Learning]
---

### Preview 

I've been thinking recently about what makes time series forecasting problems so difficult compared to other sequence learning tasks or IID Machine Learning problems. 

First some motivating baselines illustrating the difficulty of the forecasting problem. I ran a series of models
* statistical (Naive, AutoARIMA, Theta, MSTL, Seasonal-Naive)
* linear/transformer neural (DLinear, NLinear, PatchTST)
* a gradient-boosted tree (LightGBM)
* zero-shot foundation models (Chronos, TimesFM, TTM)
* "AI" LLM prompted to continue numeric sequence (Claude Opus, Haiku)

against the benchmark datasets
* m4_hourly, m4_daily
* etth
* exchange
* electricity
* bitcoin

{% include forecast_interactive.html %}

*These are interactive plots, click legend to hide a line, double click to isolate. Note that the historical sections have been truncated to 3x the forecasting window, for better visualisation.*

**Results**

There are many lines to look at, but the benchmark study's main takeaway is that for series with strong, stable seasonality such as m4 hourly, the simple statistical baselines (Seasonal-Naive, MSTL) and the zero-shot foundation models win by large margins. However, the sophisticated models tend to be completely off for many time series tasks and are not much better than naive predictors because their predictions may trend in the completely wrong direction.

We probably have some intuitions like, the forecast doesnt take into account causal information which is necessary for prediction, especially with real world data like exchange rates. But even with those "features" people are not naively making a killing in the market. 

Also, for many ML problems, we don't have full information about the casual factors either yet are able to do something reasonable, or at least better than a random walk. Hence, this post investigates the nature of the forecasting problem and what makes it so much more difficult than classical machine learning. 

*What's going on here?* 


### Why Time Series (Forecasting) is hard

<u>Problem Definition </u>

Forecasting is the act of estimating the future values of a random variable (or a set of variables) based on a historical sequence of observations, $y_{1:t}$. The goal is to learn a forecasting function $f$, which predicts the value of the variable at a future time $t + h$, where $h \ge 1$ is called the forecast horizon. 

$$\hat{y}_{t+h} = f(y_{1:t})$$

There are several variations of this problem definition, if predicting with 
* exogenous variables, then we have $f(y_{1:t}, x_{1:t})$, the forecast also conditions on external drivers $x$ (weather, promotions, prices) observed alongside the series.
* probabilistic forecast (instead of point forecast), then $p(y_{t+h} \mid y_{1:t})$. 
* multiple series, then $f(y_{1:t}, v_{1:t}, \cdots)$. This is the setting for foundation models.


<br>
#### **Time Series data comes from a Data Generating Process and is not iid**

With today's state of ML, we have very powerful and established algorithms (the deep learning enterprise) for learning a complex non-linear predictor $f_\theta: X \rightarrow Y$, where $(x, y) \sim p(x, y)$, for both classification and next-token prediction, by maximium likelihood estimation training, which minimizes expected loss over the training dataset: 

$$
\begin{equation}
\min_\theta \mathbb{E}_{(x, y)\sim p(x, y)} [\ell (f_\theta (x), y)] \approx \min_\theta \frac{1}{n}\sum_{i=1}^n [\ell (f_\theta (x_i), y_i)]
\end{equation}
$$

However, time series differs fundamentally from this setup, because of the relationship between signal and predicion in the data. In time series, we are observing a single realization/trajectory through time over $t$ time steps, not $n$ samples. In contrast, i.i.d machine learning gets $n$ independent samples from $p$, the data generating distribution. 

**Time series data comes from a Data Generating Process rather than a Data Generating Distribution**, and the crucial implication is that the sequence of data is a **joint probability distribution** over the entire path, not individual iid data points. 

\begin{equation}
p(y_1, y_2, \cdots, y_t) = p(y_1) \prod_{i=2}^t p(y_i | y_{i-1}, y_{i-2}, \cdots, y_1)
\end{equation}

This factorization holds for *any* joint distribution. But under the iid assumption of classical ML framing, conditioning on the past is dropped which greatly simplifies the prediction problem.

A lot of our learning assumptions from eq(1) do not hold and time series introduces unique challenges for learning. In particular, 

1. Time Series is Low signal to Noise (feature poor)
2. Time Series has a "lack of data"
3. Time Series suffers much more from distribution shift


<u>1. Time Series is low signal to noise (feature poor)</u>

One major difference to machine learning problems, is that in time series, the signal is scarce $f(y_t \mid y_{<t})$ whereas in i.i.d. ML the answer lives in a rich feature space where we predict $f(y_n \mid x_n)$.

Interestingly, language modeling is also sequential autoregressive prediction problem but is considered close to "solved" by LLMs. The difference is that language has high semantic content and the words (tokens) have high mutual information between the context and next token. Whereas real-valued sequences have little signal.


<u>2. Time Series has "lack of data"</u>

This is an unintuitive statement, because don't we have swaths of time series data from various sources sometimes at milisecond measurement? Maybe we don't have as much time series data as text and images on the internet, but still it doesn't seem little. The issue is that 

* *high auto-correlation shrinks the effective sample size*. If we have correlated data in a series of T timesteps, our effective sample size is less than $T$ because the T timesteps do not carry T independent pieces of information. In learning to predict $y_{t+1}$, we did not learn much more than $y_t$.

* The effective sample size that matters, is the cycles, not the timestamps. i.e., the time series realisations that result in non-autocorrelated predictions. If we were learning the patterns leading up to recession, then we only have the $m$ previous recessions as our effective sample size.

This is observable from the benchmark: m4-hourly is thousands of short series that each contribute a full daily cycle, so the *effective* sample size is far larger than any single series' length, which makes it exhibit strong seasonality and is learnable, while a single long exchange-rate path offers only a handful of independent examples to learn from.


<u>3. Time Series suffers much more from distribution shift</u>

* In ML problems, the issue of distribution shift is usually only seen after some time. For time series, the issue of distribution shift is seen during training and inference itself because of the temporal nature of the task.

* In particular, forecasting is extrapolation, not interpolation. The time index $t$ only takes values $> T$ at test, and is vulnerable to any changes in the underlying dynamics. In the classical machine learning setup, the test point might also be out of distribution, but at least it is not *guaranteed* to be.

In exchange and bitcoin which are event-driven, continually-drifting series, sophisticated models perform no better than Naive, because every forecast is an extrapolation into a distribution shifted sequence.


#### **What can we actually learn?**

Low Signal to noise, lack of data, and distribution shift are all *consequences* of the Data Generating Process (one dependent path instead of many i.i.d. draws). Since there are so many challenges with time series, what can we expect to learn at all?

*What makes a single process learnable?*

The stochastic data generating process must have three key properties (stationarity, ergodicity, auto-correlation), which allows us to estimate the process properties from the realized sequence (sampling data from the process). 

<u>Stationarity</u>. If a process is stationary, then its joint distribution is invariant to time shifts. A strict definition of stationarity says that the joint distribution of a partial sequence in the time series is the same for every time lag/shift ($h$).

\begin{equation}
p(y_1, \cdots, y_n) = p(y_{1+h}, \cdots, y_{n+h}), \quad \forall h
\end{equation}


Weak/covariance: constant mean, constant variance, and $\mathrm{Cov}(y_t, y_{t+h})$ depends only on the lag $h$, not on absolute time $t$. 

In practice, real data is never stationary, and therefore many modelling approaches also apply transformations to try and approximate stationarity.

<u>Ergodicity</u>. We only have one single realization of the data generating process, but in theory there couldl be $m$ realizations drawing from the process which we had never observed. 

A process is ergodic (for the mean) if the time average converges to the *ensemble average* (of the $m$ hypothetical realizations). 

$$\lim_{T \rightarrow \infty} \frac{1}{T}\sum_{t=1}^{T} y_t = \mathbb{E}[y_t] = \mu.$$

Without this property, no amount of data in the single realization tells us about the process, even if the process is stationary. The tricky thing about ergodicity is that it is an untestable assumption, since we never actually observed the $m$ hypothetical realizations of the time series to calculate the ensemble average.

#### **Learning the process is insufficient for forecasting.**

Having stationarity and ergodicity, means that a process' statistical properties can be deduced from a sufficiently long sample from the process. However, having the statistical properties of a data generating process does not mean that it is forecastable. 

<u>The case of White noise</u>

White noise is both stationary and ergodic, but is a series with zero autocorrelation. There is insufficient signal to predict the next value even though we know the statistical properties of white noise, the best we can do is predict the next value is the mean.

{% include whitenoise_interactive.html %}


#### **Getting predictive value from Autocorrelation**

Autocorrelation is a property of the time series data, and measures the correlation of a signal with a delayed copy of itself. Intuitively, we want to make use of **all autocorrelation signals** at every time lag and interval. In the pre-neural, purely statistical era, autocorrelation was captured through a series of intentional transformations that need to be computed. Starting from 

1. Raw autocorrelation between two time points
2. Centering (normalization)
3. Generalizing to time lags
4. Extending to multiple time points


<u>1. Raw Autocorrelation</u> 

Autocorrelation between two time points $t_1, t_2$, is the plain product-moment between the values. 

$$R_{yy}(t_1, t_2) = \mathbb{E}[y_{t_1} \times y_{t_2}]$$

This is the signal-processing definition. It is a function of *two absolute times*, and because it is not centered it mixes in the level $\mu$ — so it scales with the signal's magnitude and mean instead of isolating co-movement.

<u>2. Centering for auto-covariance </u>

Centering (normalization) by subtracting the means, measures the co-movement of *deviations* rather than absolute size:

\begin{align}
K_{yy}(t_1, t_2) &= \mathbb{E}[(y_{t_1} - \mu_{t_1}) \times (y_{t_2} - \mu_{t_2})] \\
&= \mathbb{E}[y_{t_1} \times y_{t_2}] - \mu_{t_1}\mu_{t_2}
\end{align}

<u>3. Generalizing to time lags</u>

Weak stationarity assumes that the covariance depends only on the time gap $h=t_2 - t_1$, not at the absolute time positions, so the covariance becomes a single argument function of the lags

$$\gamma (h) = K_{yy}(t_m, t_n)$$ 

for all time stamps which have the same time lag $h$, and where $h=0, 1, \cdots, T$.

<u>4. Standardise this to the Autocorrelation Function</u>

Because we are still using the raw data's scale, we should normalize it to better interpret if it is a strong or weak relationship. Dividing by the total variance of the series $\gamma(0) = \mathrm{Var}(y_t)$ to put every lag on a $[-1, 1]$ scale, comparable across lags and across series. 

$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{\mathbb{E}[(y_t - \mu)(y_{t+h} - \mu)]}{\mathbb{E}[(y_t - \mu)^2]} \in [-1, 1]$$


**How different models use the lags**

Given the ACF is a whole function over lags, different model families differ along two axes: which lags they read, and what kind of dependence they exploit at those lags.

<u>A classic linear forecasting model</u>

For instance, consider an Autorgressive AR(p) model. It forecasts the next time step by a weighted sum of the previous observed timestamps

$$\hat{y}_{t+1} = \sum_{i=1}^p \phi_i \, y_{t-i+1} = \phi_1 y_t + \phi_2 y_{t-1} + \cdots + \phi_p y_{t-p+1}$$

and to find the optimal weights ($\phi_1, \cdots, \phi_p$), we need to solve a system of equations (the **Yule–Walker equations**) that relies on the Autocorrelation Functions. 


<br>
#### **Learning structure from Multiple Processes**

*But can't we just learn over multiple time series data?*

In real world time series problems, even if we have $m$ Time-series sequences, these are considered $m$ related but separate processes, not $m$ sequences drawn from *one* process. For instance, if we consider financial markets, each individual stock price trajectory is its own process. 

We could try to learn "global" patterns, and that is exactly what Time series Foundation models (timesfm, chronos, ttm) attempt to do by training across millions of series. The benchmark shows that they help where classical seasonal baselines are already pretty good, (m4 hourly) and fail together with the rest of the forecasting models on series which have inherently low forecastabiltiy.

<br>

#### **A "Taxonomy"**

Different model families differ in which lags they read and what kind of dependence they exploit at those lags. Note that "autocorrelation" described above is a linear dependence of the mean; more complex forecasting methods attempt to use the entire shape and any non-linear relationships of the dependence.  

- Naive: just $h=1$ — carries the last value forward.
- Seasonal naive: just one lag, $h=m$ (the season length, e.g. 24 for hourly).
- AR($p$) / ARIMA: the first $p$ lags, $h=1\dots p$, as a weighted sum.
- HAR (for volatility): three fixed lags — 1, 5, 22 days (daily / weekly / monthly).
- DLinear / NLinear: a learned weight on *every* lag in the lookback window $h=1\dots L$.
- PatchTST / attention, LightGBM: also the whole window, but they can pick out and combine specific lags nonlinearly. e.g. the effect of lag 1 depends on the value at lag 2.
- Time series foundation models (timesfm, chronos, ttm): the whole window again, but with the lag-to-value mapping *pretrained* across millions of series rather than fit on the single target series.

#### **Measuring Forecastability**

Forecastability can be estimated in several ways

* Direct comparison against naive baseline, if slightly more sophisticated models don't perform any better than naive baseline, there is likely limited structural autocorrelation to take advantage of.
* Spectral forecastability, the `entropy` feature in Nixtla's `tsfeatures`. The spectral density underneath comes from `scipy.signal.periodogram` / `welch`. This measures how peaked its frequency content is.
* Ljung-box test, how much structure is there after differencing against the naive baseline. `statsmodels.stats.diagnostic.acorr_ljungbox(np.diff(y), lags=[m])` 


<br>

#### **Conclusion**

Time series is a single dependent path from a data generating *process*, not a bag of iid draws from a distribution, and with that structural difference, makes signal scarce, effective sample size small, and the test set guaranteed to lie outside the training support (extrapolation under drift). 

The main takeaway is that with this understanding, throwing more complex non-linear models (neural networks), feature engineering on the historical sequence, or pooling data using foundation models, or generation of synthetic time series data, is not where the lift is going to come from, for real world time series problems with a low measure of forecastability.

Given the lack of forecasting signal, the obvious next step then is to seek out external (exogenous) features in the real world that can help prediction models. After all many real world time series are event-driven (FX, bitcoin) and hence they are exposed to shocks and drifts which can also be measured or accounted for in the data generating process.

