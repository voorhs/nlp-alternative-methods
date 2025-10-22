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

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

---

## Outline

- Linear Attention
- State-Space Models: LSSL, S4, DSS, S4D
- Hungry Hungry Hippos (H3)
- Mamba

---

# Linear Attention

Katharopoulos et al,, "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", ICML 2020

---

## Linear Attention: Approximation of Transformer Attention

Transformer attention $\text{softmax}(QK^T)V$ can be viewed as the following:
$$
O_i=\sum_{j=1}^i\underbrace{\text{sim}(Q_i,K_j)\over\sum_{t=1}^i\text{sim}(Q_i,K_t)}_{\alpha_{ij}}V_j={\sum_{j=1}^i\text{sim}(Q_i,K_j)V_j\over\sum_{t=1}^i\text{sim}(Q_i,K_t)}
$$
where $\text{sim}(q,k)=\text{exp}(q^Tk)$. If you choose $\text{sim}(q,k)=\phi(q)^T\phi(k)$ with some non-linear $\phi:\mathbb{R}^d\to\mathbb{R}^d$, then
$$
O_i={\sum_{j=1}^i\phi(Q_i)^T\phi(K_j)V_j\over\sum_{t=1}^i\phi(Q_i)^T\phi(K_t)}=\left[{\phi(Q_i)^T\sum_{j=1}^i\phi(K_j)V_j^T\over \phi(Q_i)^T\sum_{j=1}^i\phi(K_j)}\right]^T
$$

---

## Linear Attention: Recurrency

Let us define $S_i,Z_i$:
$$
O_i^T={\phi(Q_i)^T\overbrace{\sum_{j=1}^i\phi(K_j)V_j^T}^{S_i}\over \phi(Q_i)^T\underbrace{\sum_{j=1}^i\phi(K_j)}_{Z_i}}={\phi(Q_i)^TS_i\over\phi(Q_i)^TZ_i}.
$$
Note, that
$$
\begin{align}
S_i&=S_{i-1}+\phi(K_i)V_i^T,\\
Z_i&=Z_{i-1}+\phi(K_i).
\end{align}
$$
This gives us linear-time inference.

---

## Experiments

- speed of convergence and computational costs (synthetic tasks)
- autoregressive image generation (MNIST, CIFAR-10)
- ASR with CTC loss (WSJ dataset)

**Note:** custom 200-line CUDA kernel for forward and backward passes

---

# State-Space Models

- (LSSL) "Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers" NIPS 2021
- (S4) "Efficiently Modeling Long Sequences with Structured State Spaces" ICLR 2022
- (DSS) "Diagonal State Spaces are as Effective as Structured State Spaces", NIPS 2022
- (S4D) "On the Parameterization and Initialization of Diagonal State Space Models", NIPS 2022

---

## SSM as Sequence Mapping

$\text{SSM}:u\mapsto y$
$$
\begin{align}
x_t&=\bar A x_{t-1}+\bar B u_t\\
y_t&=Cx_t+Du_t
\end{align}
$$

![bg right:53% width:600](figures/lssl-sequence-mapping.png)

---

## Recurrency & Convolution

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

## SSM: Experiments

- very long time series classification and prediction
- language modeling (a little bit)

---

# Hungry Hungry Hippos (H3)

Fu et al., "Hungry Hungry Hippos: Towards Language Modeling with State Space Models", ICLR 2023

---

## H3: Motivation

![](figures/synthetic-lm-tasks.png)

- **Induction Head** task tests how well a model can recall content after a special token
- **Associative Recall** requires the model to remember multiple key-value pairs

---

## H3: Motivation

![](figures/synthetic-lm-eval.png)

> [Transformer] can **compare** tokens by constructing the attention matrix $QK^T$, and it can recall tokens by direct **copying** (multiplying $\text{softmax}(QK^T)$ with $V$)

---

## H3 Layer: Approximation of Transformer Attention

![bg right:40% height:500](figures/h3-layer.png)

H3 layer:
$$
Q\circ\text{SSM}_\text{diag}(\text{SSM}_\text{shift}(K)\circ V)
$$

Transformer Attention:
$$
\text{softmax}(QK^T)V
$$

Linear Attention:
$$
\phi(Q_i)^T\sum_{j=1}^i\phi(K_j)V_j^T
$$

---

## Efficient GPU IO

Implementation of $(x+100)y^3$:
```python
def func(x,y):
    return x.add(100).mul(y.pow(3))
```
Lots of transfers between SRAM and HBM (static RAM and high-bandwidth memory)!

![bg right:45% width:500](figures/memory-hierarchy.png)

---

## Flash Attention: Algorithm Outline

![center width:500](figures/flash-att.png)

---

## Flash Attention: Efficiency

![center width:400](figures/fused-kernel.png)

---

## H3: FlashConv

![center width:600](figures/flash-conv.png)

---

## Experiments

- SuperGLUE: on par with GPT-Neo and OPT
- FlashConv evaluation: 2.4x faster than Transformer with FlashAttention

---

# Mamba

Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023

---

## Mamba: Motivation

All SSMs are time-invariant:

$$
\begin{align}
x_t&=\bar A x_{t-1}+\bar B u_t\\
y_t&=Cx_t+Du_t
\end{align}
$$

In order to cope with selective copying task, they must be input-dependent:

![center height:230](figures/selective-copying.png)

---

## Mamba: Selective SSM

![center](figures/mamba-algo.png)

---

## Mamba: Efficient GPU IO

![center width:900](figures/mamba-memory-io.png)

---

## Mamba: Time Complexity

Both inference and training require $O(N)$.

*Problem.* How to parallelize training? Convolution is unavailable now.

*Solution.* Recurrency can be parallelized with **parallel scan** algorithm.

---

## Mamba: Architecture

![center width:1000](figures/mamba-architecture.png)

---

## Mamba: Experiments

- Original Mamba paper
  - submitted to ICLR 2024 and denied
- Mamba study
  - short context: Harness library (WinoGrande, HellaSwag etc)
  - long context: QA and in-context learning
  - synthetic: Phonebook and RULER benchmarks

---

## Denied Mamba

![center width:700](figures/denied-mamba.png)

---

## Pure Mamba: Short Context

![center width:1000](figures/pure-mamba-eval-1T.png)

![center width:1000](figures/pure-mamba-eval-3T.png)

---

## Mamba: Synthetic Long Context

![center width:1000](figures/pure-mamba-synthetic.png)

---

## Hybrid Mamba

![center width:1000](figures/hybrid-mamba.png)

---

## Hybrid Mamba: Short Context

![center width:1000](figures/hybrid-mamba-eval-short.png)

---

## Hybrid Mamba: Long Context

![center width:1000](figures/hybrid-mamba-eval-long.png)

---

## Hybrid Mamba: Synthetic Long Context

![center width:1000](figures/hybrid-mamba-eval-synthetic.png)

---

## Hybrid Mamba: Input Length Extrapolation

![center width:1000](figures/hybrid-mamba-extrapolation.png)

---

## Conclusions

- components of good LM:
  - language modeling
  - recalling to context
  - in-context learning
- components of fast training:
  - low-level programming
  - memory hierarchy

---

## Pros and Cons of Mamba

- pros:
  - good language modeling capabilities (low perplexity)
  - fast inference
  - extrapolates input length
- cons:
  - weak in-context learning
  - fuzzy memory
  - prompt sensitive

---

## Codestral Mamba

> We have tested Codestral Mamba on in-context retrieval capabilities up to 256k tokens. We expect it to be a great **local code assistant**!

![center width:1000](figures/codestral-mamba.png)