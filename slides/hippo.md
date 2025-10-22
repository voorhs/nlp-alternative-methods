---
marp: true
math: true
---
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

# State-Space Models

Gu et al., "**HiPPO: Recurrent Memory with Optimal Polynomial Projections**", NIPS 2020

---

# Prerequisites

Functional analysis:
- Decomposition of vector in basis
- Inner product in a function space
- Orthogonal basis of a function space
- Low-dimensional approximation

Differential equations:
- Ordinary equations
- Numerical integration
- Euler method

---

## Basis Expansion

$$
\begin{align}
&\text{vector space: }\vec a\in V(\mathbb{R}),\\
&\text{basis of space: }B=\{\vec b_1,\ldots, \vec b_N\},\\
&\text{basis expansion: }\vec a=c_1\vec b_1+\ldots c_N\vec b_N,\\
&\text{coordinate vector: }\vec c=(c_1,\ldots, c_N)
\end{align}
$$

![bg right width:500](figures/standard-basis.png)

---

## Inner Product in $V(\mathbb{R})$

If basis is orthonormal:
$$
\langle \vec b_i,\vec b_j\rangle=
\begin{cases}
1, & i=j, \\
0, & i\neq j,
\end{cases}
$$
then $\vec c=(c_1,\ldots, c_N)$ is a **"good" feature representation**:
$$
c_n=\langle\vec a,\vec b_n\rangle
$$

![bg right width:500](figures/standard-basis.png)


---

## Inner Product in $L^2$

Inner product of two functions $f,g\in L^2[-1,1]$
$$
\langle f,g\rangle=\int_{-1}^1 f(x)g(x) w(x)dx.
$$

---

## Orthonormal Basis of Function Space


<!-- | Name                  | Space                  | Weight                      | Explicit Formula                                             |
| --------------------- | ---------------------- | --------------------------- | ------------------------------------------------------------ |
| trigonometric system  | $L_2[0, 2\pi]$         | $w(x)=1$                    | $\{\exp(inx)\}_{i=-\infty}^{+\infty}$                        |
| Legendre polynomials  | $L^2[-1,1]$            | $w(x)=1$                    | $P_n(x)={1\over 2^nn!}{d^n\over dx^n}(x^2-1)^n$              |
| Laguerre polynomials  | $L^2[0,+\infty)$       | $w(x)=e^{-x}$               | $L_n(x)={e^x\over n!}{d^n\over dx^n}(e^n x^n)$               |
| Chebyshev polynomials | $L^2[-1,1]$            | $w(x)={1\over\sqrt{1-x^2}}$ | $T_n(x)=\sum_{j=0}^{\lfloor n/2\rfloor}{n\choose 2j}(x^2-1)^jx^{n-2k}$ |
| Hermite polynomials   | $L^2(-\infty,+\infty)$ | $w(x)=e^{-x^2}$             | $H_n(x)=(-1)^ne^{x^2}{d^n\over dx^n}e^{-x^2}$                | -->

![center width:950](figures/orthonormal-polynomials.png)

---

## Function Expansion

Let $f(x)\in L_2[-1,1]$. Let $\{P_n(x)\}_{n=1}^\infty$ be an orthonormal basis w.r.t. $w(x)$. Then the **optimal approximation** is the following:
$$
\begin{align}
f(x)&=\sum_{n=1}^\infty c_n P_n(x)\\
&\approx \sum_{n=1}^N c_n P_n(x).
\end{align}
$$

Coefficients $c=(c_1,\ldots,c_N)$ form a **feature representation**:
$$
c_n=\langle f(x),P_n(x)\rangle=\int\limits_{-1}^{1} f(x)P_n(x)w(x)dx.
$$

---

## Ordinary Differential Equations

ODE describes the dynamics of a system:
$$
{d\over dt}x(t)=f(x(t),t).
$$

Solving ODE (numerically or analytically) is called **integration**.

![bg right:40% width:450](figures/butterfly.png)

---

## Forward Euler Method for Numerical ODE Integration
$$
\begin{align}
{x(t+dt)-x(t)\over dt}&=f(x(t),t),\\
x(t+dt)&=x(t)+f(x(t),t)dt,\\
x_{k+1}&=x_k+f(x_k,k)dt.
\end{align}
$$
$\text{d}t$ is a hyperparameter called **discretization step**.

---

## ODE Integration Example

Copter dynamics:

$$
\begin{cases}
    {d\over dt} x= v_{x} \\
    {d\over dt} v_x= - \frac{(u_1 + u_2)\sin(\theta)}{m} \\
    {d\over dt} y= v_{y} \\
    {d\over dt} v_y= \frac{(u_1 + u_2)\cos(\theta)}{m} - g\\
    {d\over dt}\theta = \omega \\
    {d\over dt}\omega = \frac{(u_1 - u_2)r}{I}
\end{cases}
\Rightarrow
\begin{cases}
    x_{n+1} = x_{n} + v_{n,x}\text{d}t \\
    v_{n+1,x} = v_{n,x} - \frac{(u_1 + u_2)\sin(\theta)}{m}\text{d}t \\
    y_{n+1} = y_{n} + v_{n,y}\text{d}t \\
    v_{n+1,y} = v_{n,y} + \left(\frac{(u_1 + u_2)\cos(\theta)}{m} - g\right)\text{d}t \\
    \theta_{n+1} = \theta_{n} + \omega_{n}\text{d}t \\
    \omega_{n+1} = \omega_{n} + \frac{(u_1 - u_2)r}{I}\text{d}t
\end{cases}
$$

---

# HiPPO: Key Concepts

- Univariate time series
- Univariate continuous function
- Expansion coefficients as memory


---
## Univariate Function

![center width:850](figures/continuous-function-1-01.drawio.svg)

---

## Expansion Coefficients as Memory

![center width:850](figures/continuous-function-1-02.drawio.svg)

---

## Expansion Coefficients as Memory

![center width:850](figures/continuous-function-1-03.drawio.svg)

---

## Recurrent Memory as ODE Dynamics

![center width:850](figures/continuous-function-1-04.drawio.svg)

---

## Numerical Solution of ODE

![center width:850](figures/continuous-function-1-05.drawio.svg)

---

# Some Details

- Translated polynomial basis
- Sliding window polynomial basis
- Intuition behind weight functions


---

## Orthonormal Basis for Function Space

![center width:950](figures/orthonormal-polynomials.png)

---

## How to expand function on $[0,\tau]$ using basis for $[-1,1]$?

Use linear mapping of the argument!
$$P_n(x)\mapsto P_n(ax+b)$$

---

## Translated Polynomial Basis

![center width:850](figures/continuous-function-1-06.drawio.svg)

---

## How not to pay attention to old history?

Use sliding window and weight functions!

---

## Sliding Window Polynomial Basis

![center width:850](figures/continuous-function-1-07.drawio.svg)

---

## Sliding Window Polynomial Basis

![center width:850](figures/continuous-function-1-08.drawio.svg)

---

## Weight Functions

![center height:300](figures/weight-function.png)

---

# Instantiations of HiPPO

- Translated Legendre
- Translated Laguerre
- Scaled Legendre

---

## Translated Legendre (LegT)

- Space: $L^2[\tau-\theta,\tau]$
- Basis: Legendre Polynomials
- Weight function: $w(t)={1\over\theta}[\tau-\theta\leqslant t\leqslant \tau]$

$$
A_{nk}={1\over\theta}
\begin{cases}
(-1)^{n-k}(2n+1), & n\geqslant k, \\
2n+1, & n\leqslant k, \\
\end{cases}\qquad
B_n={1\over\theta}(2n+1)(-1)^n.
$$

---

## Translated Laguerre (LagT)

- Space: $L^2[-\infty,\tau]$
- Basis: Laguerre Polynomials
- Weight function: $w(t)=\exp(t-\tau)[t\leqslant \tau]$

$$
A_{nk}={1\over\theta}
\begin{cases}
1, & n\geqslant k, \\
0, & n< k,
\end{cases}\qquad
B_n=1.
$$

---

## Scaled Legendre (LegS)

- Space: $L^2[0,\tau]$
- Basis: Laguerre Polynomials
- Weight function: $w(t)={1\over\tau}[0\leqslant t\leqslant\tau]$

$$
A_{nk}=-{1\over\tau}
\begin{cases}
\sqrt{(2n+1)(2k+1)}, & n> k, \\
n+1, & n=k, \\
0, & n< k
\end{cases}\qquad
B_n=\sqrt{2n+1}.
$$

---

# Experiments

- Integration with RNN
- pMNIST
- Copying
- Trajectory Classification
- IMDB Review Dataset: Text classification
- Mackey Glass Prediction: time series prediction, predict next 15 steps

---

## Integration with RNN

Basic gated RNN:
$$
\begin{align}
\text{RNN}(h,x)&=(1-g)\circ h + g\circ\tanh(W_1h+U_1x+b_1),\\
g&=\sigma(W_2h+U_2x+b_2)
\end{align}
$$

![bg right:30% width:300](figures/rnn-cell.drawio.svg)

---

## Integration with RNN

With HiPPO:
$$
\text{RNN}(h, [c, x]),
$$
where
$$
c_t=A_tc_{t-1}+b_tf_t.
$$
i.e. HiPPO coefficients of $f(t)=w^Th_t$, where $w$ is trained.

![bg right:40% width:400](figures/rnn-cell-hippo.drawio.svg)

---

## pMNIST: Univariate Time Series

SOTA!!

![center height:150](figures/pmnist.png)

![bg right:40% width:400](figures/pmnist-metrics.png)

---

## Copying

> $L + 20$ digits where the first 10 tokens $(a_0 , a_1 \ldots a_9 )$ are randomly chosen from $\{1, . . . , 8\}$, the middle $N$ tokens are set to $0$, and the last ten tokens are $9$. The goal of the recurrent model is to output $(a_0 \ldots a_9)$ in order on the last $10$ time steps.

No metrics reported

---

## Trajectory Classification

Data sample:

![center width:300](figures/char-traj.png)

Time series with three variables: $x,y$ and pressure

---

## Trajectory Classification

![center height:250](figures/char-traj-metrics.png)

---

## Speed & Reconstruction

> The input function is randomly sampled from a continuous-time band-limited **white noise process**, with length $10^6$. The sampling step size is $\Delta t = 10^{−4}$ , and the signal band limit is 1Hz.

![center width:500](figures/speed-of-reconstruction.png)

---

## Sentiment Classification on the IMDB Movie Review Dataset

Finally Language Modeling

![center width:500](figures/imdb-metrics.png)

---

## Mackey Glass Prediction

> **time series prediction** task for modeling chaotic dynamical systems... The data is a sequence of one-dimensional observations, and models are tasked with predicting **15 time steps into the future**.

![center width:550](figures/mackey-metrics.png)

---

# Conclusion

- expansion coefficients as memory
- fast recurrent inference as ODE integration
- window size and weight functions for paying attention
- Attempt of using with NN: Integration with RNN (potential is not fully expored)
- SOTA on time series tasks
- Limited research on language modeling

---

# Next Seminar: LSSL

Linear state-space layer!

![center width:700](figures/continuous-function-1-lssl.drawio.svg)