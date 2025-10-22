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

- (LSSL) "Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers" NIPS 2021
- (S4) "Efficiently Modeling Long Sequences with Structured State Spaces" ICLR 2022
- (DSS) "Diagonal State Spaces are as Effective as Structured State Spaces", NIPS 2022
- (S4D) "On the Parameterization and Initialization of Diagonal State Space Models", NIPS 2022

---

## HiPPO: Expansion Coefficients as Memory

![center width:850](figures/continuous-function-1-02.drawio.svg)

---

## HiPPO: Expansion Coefficients as Memory

![center width:850](figures/continuous-function-1-03.drawio.svg)

---

## HiPPO: Recurrent Memory as ODE Dynamics

![center width:850](figures/continuous-function-1-04.drawio.svg)

---

## HiPPO: Numerical Solution of ODE

![center width:850](figures/continuous-function-1-05.drawio.svg)

---

## HiPPO: Scaled Legendre (LegS)

- Space: $L^2[0,\tau]$
- Basis: Laguerre Polynomials
- Weight function: $w(t)={1\over\tau}[0\leqslant t\leqslant\tau]$

$$
A_{ij}=-{1\over\tau}
\begin{cases}
\sqrt{(2i+1)(2j+1)}, & i>j, \\
i+1, & i=j, \\
0, & i<j
\end{cases}\qquad
B_i=\sqrt{2i+1}.
$$

---

## ODE Integration: Euler method

$$
\begin{align}
{d\over dt}c(t)&=Ac(t)+Bf(t)\\
c_{n+1}&=c_n+(Ac_n+Bf_n)dt\\
&=(I+Adt)c_n+(Bdt)f_n\\
&=\bar A c_n+\bar B f_n.
\end{align}
$$

Let us call $\bar A=I+Adt$ **discretized** version of $A$. Same for $\bar B=Bdt$ and $B$.

---

## ODE Integration: Other Methods

Backward Euler method:
$$
\begin{align}
\bar A&=(I-Adt)^{-1}\\
\bar B&=(I-Adt)^{-1}Bdt
\end{align}
$$

Bilinear method:
$$
\begin{align}
\bar A&=\left(I-A{dt/2}\right)^{-1}\left(I+A{dt/ 2}\right)\\
\bar B&=\left(I-A{dt/2}\right)^{-1}Bdt
\end{align}
$$

Zero-order hold method (ZOH):
$$
\begin{align}
\bar A&=e^{Adt}\\
\bar B&=(\bar A-I)A^{-1}B
\end{align}
$$

---

## HiPPO: Integration with RNN

With HiPPO:
$$
\text{RNN}(h, [c, x]),
$$
where
$$
c_t=\bar Ac_{t-1}+\bar Bf_t.
$$
i.e. HiPPO coefficients of $f(t)=w^Th_t$, where $w$ is trained.

![bg right:40% width:400](figures/rnn-cell-hippo.drawio.svg)

---

# LSSL: Linear State-Space Layer

---

## Sequence Mapping

$\text{SSM}:u\mapsto y$
$$
\begin{align}
x_t&=\bar A x_{t-1}+\bar B u_t\\
y_t&=Cx_t+Du_t
\end{align}
$$

(Optionally) trained:
- $\bar A\in\mathbb{R}^{N\times N}$, $\bar B\in\mathbb{R}^{N}$.

Trained:
- $C\in\mathbb{R}^{M\times N}$, $D\in\mathbb{R}^M$.

It means, that $\text{SSM}:\mathbb{R}\to\mathbb{R}^M$.

![bg right:53% width:600](figures/lssl-sequence-mapping.png)

---

## LSSL: Recurrency = Convolution

Unroll recurrency:
$$
\begin{align}
y_t&=Cx_t+Du_t\\
&=C(\bar A x_{t-1}+\bar B u_t)+Du_t\\
&=C(\bar A (\bar A x_{t-2}+\bar B u_{t-1})+\bar B u_t)+Du_t\\
&=\ldots\\
&=C(\bar A)^t\bar Bu_0+C(\bar A)^{t-1}\bar Bu_1+\ldots+C\bar A\bar B u_{t-1}+C\bar Bu_t+Du_t
\end{align}
$$

Thus, SSM is convolution:
$$
y=\mathcal{K}_L(\bar A,\bar B,C)* u+Du.
$$
with kernel
$$
\mathcal{K}_L(A,B,C)=(CB, CAB, \ldots, CA^{L-1}B)\in\mathbb{R}^{M\times L}
$$

<!-- Because $CA^tB=(M\times N)\cdot (N\times N)\cdot (N\times 1)=M\times 1$. -->

---

## Time Complexity

- Training: $O(L\log L)$ thanks to FFT convolution 
- Inference: $O(L)$ thanks to recurrency

---

(Optionally) trained:
- $\bar A\in\mathbb{R}^{N\times N}$,
- $\bar B\in\mathbb{R}^{N}$.

Trained:
- $C\in\mathbb{R}^{M\times N}$,
- $D\in\mathbb{R}^M$.

![bg left:65% width:700](figures/continuous-function-1-lssl.drawio.svg)

---

## Tridiagonal Parametrization

**Theorem.** The class of $N\times N$ matrices $\mathcal{S}_N = \{P (D + T^{-1})Q\}$ with diagonal $D$, $P$, $Q$ and tridiagonal $T$ includes the original HiPPO-LegS, HiPPO-LegT, and HiPPO-LagT matrices.

**It means, that one can learn $A$ as $6N$ parameters instead of $N^2$.**

![bg right:40% width:400](figures/learning-a-t-ablations.png)

---

## Experiments: Very Long Time Series

- Sequence image classification (sMNIST, pMNIST, sCIFAR, CelebA)
- Audio waveform classification (1-sec Speech Commands)
- Modeling and Computational Benefits of LSSLs

---

## DL Audio

![center](figures/spec-vs-waveform.png)

---

## Modeling and Computational Benefits of LSSLs

![](figures/lssl-benefits.png)

---

## Problem of LSSL

Computation of $\mathcal{K}_L(\bar A,\bar B,C)=(C\bar A^k\bar B)_{k=0}^{L-1}$ must be efficient!
- naive: $O(LN^2)$ operations and $O(LN)$ memory
- LSSL: $\widetilde O(N+L)$ operations and $O(L)$ memory **(in theory)**
- S4, DSS, S4D: $\widetilde O(N+L)$ operations and $O(N+L)$ memory

where $N$ is number of polynomials in HiPPO, $L$ is length of sequence.

---

## Simplify Structure of $\bar A$

Kernel needs powers of $\bar A$:
$$
\mathcal{K}_L(\bar A,\bar B,C)=(C\bar B, C\bar A\bar B, \ldots, C\bar A^{L-1}\bar B)\in\mathbb{R}^{M\times L}
$$

Methods:
- LSSL suggests **tridiagonal parametrization**
- S4 suggests **NPLR parametrization** for bilinear integration
- DSS suggests **diagonal parametrization** for ZOH integration
- S4D suggests **diagonal parametrization** for ZOH and bilinear

---

# S4: Structured State-Space Sequence Model

---

## S4 Parameterization

The class of "Normal Plus Low-Rank" matrices $A=V\Lambda V^*-PQ^*$ with diagonal $\Lambda$, unitary $V$ and low-rank matrices $P,Q\in\mathbb{R}^{N\times r}$ ($3N$ trained parameters).

![center width:1080](figures/s4-kernel.png)

---

## S4 Initialization (Hippo)

Let us use NPLR representation of HiPPO matrices:
$$
A=V\Lambda V^*-PQ^T=V(\Lambda-(V^*P)(V^*Q)^*)V^*,
$$
where $V\in\mathbb{C}^{N\times N}$ is unitary, $\Lambda=\text{diag}(\lambda_1,\ldots,\lambda_N)$, $P,Q\in\mathbb{R}^{N\times r}$.

Then, one can use these $\Lambda, V^*P, V^*Q$ as initialization for $A=\Lambda-PQ^*$ (without any change of basis for input $u$?).

---


## Experiments (Text, Images, Audio)

- Time series classification (**Long-range arena**)
- Audio waveform classification (1-sec speech commands)
- Autoregressive generation (CIFAR-10, **WikiText-103**)
- Time series forecasting (weather etc)

---

## Long-Range Arena

![center width:900](figures/lra-evolution.svg)

---

## Causal Language Modeling

![center width:600](figures/s4-clm.png)

---

# DSS: Diagonal State Space Model

---

## Problem of S4

> ...need to employ several reduction steps and linear algebraic techniques to be able to compute the state space output efficiently, **making S4 difficult to understand, implement and analyze**.

---

## DSS Parameterization

If $A=V\Lambda V^{-1}$, then $\exists w\in\mathbb{C}^N$:
$$
\mathcal{K}_L(\bar A, \bar B, C)\Leftrightarrow \mathcal{K}_L(\Lambda, (e^{L\lambda_i dt}-1)_{i=1}^N, w)=w\Lambda^{-1}\text{softmax}(P),
$$
where $P\in\mathbb{C}^{N\times L}$ such that $P_{ij}=\lambda_ij\cdot dt$.

![center width:1080](figures/dss-kernel.png)

---

## DSS Initialization (Skew-Hippo)

Same trick from S4:
> Let us use NPLR representation of HiPPO matrices:
> $$
> A=V\Lambda V^*-PQ^T=V(\Lambda-(V^*P)(V^*Q)^*)V^*,
> $$
> where $V\in\mathbb{C}^{N\times N}$ is unitary, $\Lambda=\text{diag}(\lambda_1,\ldots,\lambda_N)$, $P,Q\in\mathbb{R}^{N\times r}$.
> Then, one can use these $\Lambda, V^*P, V^*Q$ as initialization for $A=\Lambda-PQ^*$.

Then, one can use $\Lambda$ as initialization for $A=\Lambda\in\mathbb{C}^{N\times N}$ (without any change of basis for input $u$?).

---

## Experiments

- Time series classification (long-range arena)
- Audio waveform classification (1-sec speech commands)
- Kernel visualization

---

## Kernel Visualization

![center width:1000](figures/dss-kernel-vis.png)


---

## Problem of DSS

- Still too complicated!
- Not explained theoretically

---

# S4D: Diagonal Approximations of S4

---

## Vandermonde Matrix

Let $x\in\mathbb{R}^{m}$. Vandermonde matrix is
$$
\mathcal{V}(x)=
\begin{pmatrix}
1 & x_0 & x_0^2 & \ldots & x_0^n \\
1 & x_1 & x_1^2 & \ldots & x_1^n \\
1 & x_2 & x_2^2 & \ldots & x_2^n \\
\vdots & \vdots & \vdots & \ddots & \vdots &  \\
1 & x_m & x_m^2 & \ldots & x_m^n \\
\end{pmatrix}
$$

> The discrete Fourier transform is defined by a specific Vandermonde matrix, the DFT matrix, where the $x_{i}$ are chosen to be nth roots of unity. The **Fast Fourier transform** computes the product of this matrix with a vector in $O(n\log^2n)$ time.

---

## S4D Parameterization

Constrain real part to be negative and $A$ to be diagonal:
$$
A=-\exp(A_\text{Re})+iA_\text{Im}.
$$

Kernel computation is a simple matrix-vector computation:
$$
K_k =\sum_{i=0}^{N-1}C_i\bar A_i^k\bar B_i\Rightarrow K = (\bar B^T \circ C)\mathcal{V}_L (\bar A),
$$
where $\mathcal{V}_L(A)_{ik} = A_{i}^k$ is Vandermonde matrix made out of $\text{diag}(A)$.

---

## S4D Initialization (Proxy Hippo)

S4D-Inv:
$$
A_{nn}=-{1\over2}+i{N\over \pi}\left({N\over 2n+1}-1\right).
$$

S4D-Lin:
$$
A_{nn}=-{1\over2}+i\pi n.
$$

---

# Summary

- HiPPO: polynomial projections as recurrent memory
- LSSL: recurrent inference and conv training
- LSSL: you can learn $A$ (tridiagonal parametrization)
- S4, DSS, S4D: you can learn $A$ efficiently (diagonal parametrization)
- HiPPO initialization is extremely good
- SOTA time series models
- application to language modeling still isn't explored