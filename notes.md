# State-Space Models: From HiPPO to Mamba

## Abstract

This lecture presents a comprehensive overview of State-Space Models (SSMs) in deep learning, tracing their evolution from the foundational HiPPO framework to the modern Mamba architecture. We begin with the mathematical foundations of polynomial projections and optimal memory mechanisms, then explore how these concepts were adapted into efficient sequence modeling architectures. The lecture covers three main developments: HiPPO's recurrent memory with optimal polynomial projections, the transition to structured state-space layers (LSSL, S4, DSS, S4D), and finally the selective state-space models exemplified by Mamba. Throughout, we examine both theoretical foundations and practical implementations, highlighting how these models achieve linear-time complexity while maintaining competitive performance with Transformers on various sequence modeling tasks.

## Table of Contents

1. [Mathematical Prerequisites](#mathematical-prerequisites)
2. [HiPPO: Recurrent Memory with Optimal Polynomial Projections](#hippo-recurrent-memory-with-optimal-polynomial-projections)
3. [State-Space Models: From Theory to Practice](#state-space-models-from-theory-to-practice)
4. [Mamba: Selective State Spaces](#mamba-selective-state-spaces)
5. [Conclusion](#conclusion)

---

## Mathematical Prerequisites

### Functional Analysis Foundations

Before diving into State-Space Models, we need to establish the mathematical foundations that make these architectures possible. The key concepts involve:

**Basis Expansion in Vector Spaces**

Consider a vector space $V(\mathbb{R})$ with a basis $B = \{\vec{b}_1, \ldots, \vec{b}_N\}$. Any vector $\vec{a} \in V$ can be expressed as:

$$\vec{a} = c_1\vec{b}_1 + \ldots + c_N\vec{b}_N$$

where $\vec{c} = (c_1, \ldots, c_N)$ is the coordinate vector representing $\vec{a}$ in the basis $B$.

If the basis is orthonormal:
$$\langle \vec{b}_i, \vec{b}_j \rangle = \begin{cases} 1, & i = j \\ 0, & i \neq j \end{cases}$$

then the coefficients can be computed as:
$$c_n = \langle \vec{a}, \vec{b}_n \rangle$$

**Function Spaces and Orthonormal Bases**

The same principles extend to function spaces. For functions $f, g \in L^2[-1,1]$, we define the inner product as:

$$\langle f, g \rangle = \int_{-1}^1 f(x)g(x) w(x) dx$$

where $w(x)$ is a weight function. Common orthonormal polynomial bases include:

- **Legendre polynomials**: $L^2[-1,1]$ with $w(x) = 1$
- **Laguerre polynomials**: $L^2[0,+\infty)$ with $w(x) = e^{-x}$
- **Chebyshev polynomials**: $L^2[-1,1]$ with $w(x) = \frac{1}{\sqrt{1-x^2}}$
- **Hermite polynomials**: $L^2(-\infty,+\infty)$ with $w(x) = e^{-x^2}$

**Function Approximation**

For any function $f(x) \in L^2[-1,1]$ and orthonormal basis $\{P_n(x)\}_{n=1}^\infty$, the optimal approximation is:

$$f(x) = \sum_{n=1}^\infty c_n P_n(x) \approx \sum_{n=1}^N c_n P_n(x)$$

where the coefficients form a feature representation:
$$c_n = \langle f(x), P_n(x) \rangle = \int_{-1}^1 f(x)P_n(x)w(x)dx$$

### Differential Equations and Numerical Integration

**Ordinary Differential Equations**

ODEs describe system dynamics:
$$\frac{d}{dt}x(t) = f(x(t), t)$$

Solving ODEs (numerically or analytically) is called integration.

**Forward Euler Method**

The simplest numerical integration scheme:
$$\frac{x(t+dt) - x(t)}{dt} = f(x(t), t)$$

This gives us the recurrence relation:
$$x_{k+1} = x_k + f(x_k, k)dt$$

where $dt$ is the discretization step.

**Other Integration Methods**

- **Backward Euler**: $\bar{A} = (I - Adt)^{-1}$, $\bar{B} = (I - Adt)^{-1}Bdt$
- **Bilinear method**: $\bar{A} = \left(I - A\frac{dt}{2}\right)^{-1}\left(I + A\frac{dt}{2}\right)$
- **Zero-order hold (ZOH)**: $\bar{A} = e^{Adt}$, $\bar{B} = (\bar{A} - I)A^{-1}B$

---

## HiPPO: Recurrent Memory with Optimal Polynomial Projections

### Core Concepts

HiPPO (High Order Polynomial Projections) introduces a revolutionary approach to recurrent memory by using expansion coefficients as memory states. The key insight is treating univariate time series as continuous functions and using polynomial projections to compress historical information optimally.

**The HiPPO Framework**

1. **Univariate Function Representation**: Each time series is viewed as a continuous function $f(t)$
2. **Expansion Coefficients as Memory**: Instead of storing raw history, we store coefficients $c(t)$ of polynomial expansion
3. **Recurrent Memory as ODE Dynamics**: The coefficients evolve according to differential equations
4. **Numerical Solution**: Use numerical integration to update memory states

### Mathematical Formulation

For a function $f(t)$ defined on $[0, \tau]$, we seek to approximate it using polynomial basis functions. The optimal coefficients $c(t)$ satisfy:

$$\frac{d}{dt}c(t) = Ac(t) + Bf(t)$$

where $A$ and $B$ are matrices determined by the chosen polynomial basis and integration method.

**Discretization**

Applying numerical integration (e.g., Forward Euler):
$$c_{n+1} = c_n + (Ac_n + Bf_n)dt = (I + Adt)c_n + (Bdt)f_n$$

Letting $\bar{A} = I + Adt$ and $\bar{B} = Bdt$, we get:
$$c_{n+1} = \bar{A}c_{n-1} + \bar{B}f_n$$

### HiPPO Instantiations

**Translated Legendre (LegT)**
- Space: $L^2[\tau-\theta, \tau]$ (sliding window)
- Weight function: $w(t) = \frac{1}{\theta}[\tau-\theta \leq t \leq \tau]$
- Matrix elements:
  $$A_{nk} = \frac{1}{\theta}\begin{cases}
  (-1)^{n-k}(2n+1), & n \geq k \\
  2n+1, & n \leq k
  \end{cases}$$
  $$B_n = \frac{1}{\theta}(2n+1)(-1)^n$$

**Translated Laguerre (LagT)**
- Space: $L^2[-\infty, \tau]$ (exponential weighting)
- Weight function: $w(t) = \exp(t-\tau)[t \leq \tau]$
- Matrix elements:
  $$A_{nk} = \begin{cases}
  1, & n \geq k \\
  0, & n < k
  \end{cases}$$
  $$B_n = 1$$

**Scaled Legendre (LegS)**
- Space: $L^2[0, \tau]$ (full history)
- Weight function: $w(t) = \frac{1}{\tau}[0 \leq t \leq \tau]$
- Matrix elements:
  $$A_{nk} = -\frac{1}{\tau}\begin{cases}
  \sqrt{(2n+1)(2k+1)}, & n > k \\
  n+1, & n = k \\
  0, & n < k
  \end{cases}$$
  $$B_n = \sqrt{2n+1}$$

### Integration with Neural Networks

HiPPO was integrated with RNNs by treating the expansion coefficients as additional context:

$$\text{RNN}(h, [c, x])$$

where $c_t = \bar{A}c_{t-1} + \bar{B}f_t$ represents HiPPO coefficients of $f(t) = w^T h_t$ (with learnable $w$).

### Experimental Results

HiPPO achieved state-of-the-art results on several benchmarks:

- **Permuted MNIST**: SOTA performance on sequence image classification
- **Trajectory Classification**: Robust performance across different sampling rates
- **Mackey Glass Prediction**: Superior long-term prediction capabilities
- **IMDB Sentiment**: Competitive performance on text classification

The key advantages were:
- Fast recurrent inference through ODE integration
- Optimal memory compression via polynomial projections
- Theoretical guarantees for approximation quality

---

## State-Space Models: From Theory to Practice

### Linear State-Space Layers (LSSL)

Building on HiPPO, LSSL transforms the theoretical framework into a practical neural network layer.

**Sequence Mapping**

An SSM maps input sequence $u$ to output sequence $y$:
$$\begin{align}
x_t &= \bar{A} x_{t-1} + \bar{B} u_t \\
y_t &= Cx_t + Du_t
\end{align}$$

where:
- $\bar{A} \in \mathbb{R}^{N \times N}$, $\bar{B} \in \mathbb{R}^N$ (optionally trainable)
- $C \in \mathbb{R}^{M \times N}$, $D \in \mathbb{R}^M$ (trainable)

**Recurrency as Convolution**

Unrolling the recurrence reveals that SSMs implement convolution:
$$y_t = C(\bar{A})^t\bar{B}u_0 + C(\bar{A})^{t-1}\bar{B}u_1 + \ldots + C\bar{B}u_t$$

This can be written as:
$$y = \mathcal{K}_L(\bar{A}, \bar{B}, C) * u + Du$$

where the kernel is:
$$\mathcal{K}_L(A, B, C) = (CB, CAB, \ldots, CA^{L-1}B) \in \mathbb{R}^{M \times L}$$

**Computational Complexity**

- **Training**: $O(L \log L)$ via FFT convolution
- **Inference**: $O(L)$ via recurrence

This dual nature allows efficient parallel training while maintaining fast sequential inference.

### Parameterization Strategies

**Tridiagonal Parameterization (LSSL)**

The key insight is that HiPPO matrices can be represented as:
$$A = P(D + T^{-1})Q$$

where $D$, $P$, $Q$ are diagonal and $T$ is tridiagonal. This reduces parameters from $N^2$ to $6N$ while maintaining the theoretical properties.

**Normal Plus Low-Rank (S4)**

S4 uses the parameterization:
$$A = V\Lambda V^* - PQ^*$$

where $\Lambda$ is diagonal, $V$ is unitary, and $P, Q \in \mathbb{R}^{N \times r}$ are low-rank matrices. This enables efficient kernel computation through specialized algorithms.

**Diagonal Parameterization (DSS, S4D)**

The simplest approach uses diagonal matrices:
$$A = \text{diag}(\lambda_1, \ldots, \lambda_N)$$

This allows extremely efficient kernel computation:
$$K_k = \sum_{i=0}^{N-1} C_i \bar{A}_i^k \bar{B}_i$$

### Initialization Strategies

All methods initialize from HiPPO matrices:

**S4D Initialization**
- **S4D-Inv**: $A_{nn} = -\frac{1}{2} + i\frac{N}{\pi}\left(\frac{N}{2n+1} - 1\right)$
- **S4D-Lin**: $A_{nn} = -\frac{1}{2} + i\pi n$

### Experimental Results

**Long-Range Arena**: S4 achieved breakthrough results on very long sequence classification tasks, significantly outperforming previous methods.

**Language Modeling**: Initial experiments showed competitive perplexity with Transformers while being 60x faster at generation.

**Audio Processing**: Direct processing of raw audio waveforms (16kHz sampling) without spectral preprocessing.

---

## Mamba: Selective State Spaces

### Motivation and Problem Statement

While SSMs showed promise, they suffered from a fundamental limitation: **time-invariance**. The parameters $A$, $B$, $C$ remained constant regardless of input, making it difficult to selectively remember or forget information based on context.

**Selective Copying Task**

Consider a task where the model must selectively copy tokens based on context. Traditional SSMs struggle because they cannot adapt their memory mechanism to the input content.

### Linear Attention Prelude

Before Mamba, Linear Attention showed how to approximate Transformer attention efficiently:

**Standard Attention**
$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V$$

**Linear Attention Approximation**
Replace $\text{sim}(q, k) = \exp(q^T k)$ with $\text{sim}(q, k) = \phi(q)^T \phi(k)$:

$$O_i = \frac{\phi(Q_i)^T \sum_{j=1}^i \phi(K_j)V_j^T}{\phi(Q_i)^T \sum_{j=1}^i \phi(K_j)}$$

This enables recurrent computation:
$$\begin{align}
S_i &= S_{i-1} + \phi(K_i)V_i^T \\
Z_i &= Z_{i-1} + \phi(K_i)
\end{align}$$

### Hungry Hungry Hippos (H3)

H3 bridged SSMs and language modeling by approximating attention with SSM components:

**H3 Layer**
$$Q \circ \text{SSM}_{\text{diag}}(\text{SSM}_{\text{shift}}(K) \circ V)$$

This architecture:
- Approximates Transformer attention using SSMs
- Achieves competitive results on SuperGLUE
- Uses Flash Convolution for efficient GPU utilization

### Mamba Architecture

**Selective State Spaces**

Mamba introduces input-dependent parameters:

$$\begin{align}
x_t &= \bar{A}(x_t) x_{t-1} + \bar{B}(x_t) u_t \\
y_t &= C(x_t) x_t + D(x_t) u_t
\end{align}$$

where $A$, $B$, $C$ are now functions of the input $x_t$.

**Implementation Details**

- **Parameter Functions**: Simple linear projections from input
- **Selective Mechanism**: Gates control information flow
- **Efficient Implementation**: Custom CUDA kernels for memory hierarchy optimization

**Parallel Scan Algorithm**

Since convolution is no longer available, Mamba uses parallel scan to parallelize the recurrence:
- Enables $O(N)$ parallel computation during training
- Maintains linear-time inference
- Adapts cumulative sum algorithms to SSM recurrence

### Mamba Architecture Components

**Complete Architecture**
1. **Input Projection**: Linear layer to hidden dimension
2. **Selective SSM**: Input-dependent state-space computation
3. **Gated MLP**: Modern activation functions (SiLU/Swish)
4. **Output Projection**: Back to original dimension

**Memory Hierarchy Optimization**

Mamba implements sophisticated memory management:
- **SRAM**: Fast on-chip memory for active computations
- **HBM**: High-bandwidth memory for parameter storage
- **Fused Operations**: Minimize memory transfers

### Experimental Evaluation

**Short Context Tasks**

On standard language modeling benchmarks (Harness library):
- **Competitive Performance**: On par with Transformers on most tasks
- **In-Context Learning**: Slightly weaker than Transformers
- **Scaling**: Performance gap closes with larger datasets

**Long Context Tasks**

- **Question Answering**: Competitive on long-context QA tasks
- **Synthetic Tasks**: Strong performance on Phonebook and RULER benchmarks
- **Extrapolation**: Better length extrapolation than Transformers

**Hybrid Architectures**

Combining Mamba with Transformer components:
- **Best of Both Worlds**: Combines Mamba's efficiency with Transformer's capabilities
- **Improved Performance**: Better results than pure Mamba
- **Flexible Design**: Allows mixing different architectural components

### Performance Characteristics

**Advantages**
- **Fast Inference**: Linear-time complexity
- **Memory Efficient**: Constant memory usage
- **Length Extrapolation**: Better than Transformers
- **Low Perplexity**: Competitive language modeling

**Limitations**
- **In-Context Learning**: Weaker than Transformers
- **Fuzzy Memory**: Less precise than attention mechanisms
- **Prompt Sensitivity**: More sensitive to input formatting

---

## Conclusion

The evolution from HiPPO to Mamba represents a remarkable journey in sequence modeling, demonstrating how theoretical insights can be transformed into practical architectures that challenge the dominance of Transformers.

### Key Contributions

**HiPPO**: Established the theoretical foundation using optimal polynomial projections for recurrent memory, providing a principled approach to compressing historical information.

**State-Space Models**: Bridged theory and practice by showing how recurrent dynamics can be implemented as efficient convolutions, enabling parallel training while maintaining sequential inference.

**Mamba**: Introduced selective mechanisms that allow models to adapt their memory based on input content, achieving competitive performance with linear-time complexity.

### Technical Insights

1. **Mathematical Foundations**: Polynomial projections provide optimal compression of historical information
2. **Dual Nature**: The recurrence-convolution duality enables efficient training and inference
3. **Selective Mechanisms**: Input-dependent parameters are crucial for handling complex sequence patterns
4. **Memory Hierarchy**: Careful attention to GPU memory organization is essential for practical efficiency

### Future Directions

The success of Mamba and related architectures suggests several promising directions:

- **Hybrid Architectures**: Combining the strengths of different architectural paradigms
- **Specialized Applications**: Tailoring architectures for specific domains (e.g., code generation)
- **Scaling Laws**: Understanding how these models scale with parameters and data
- **Theoretical Analysis**: Deeper understanding of why selective mechanisms work

### Practical Implications

For practitioners, State-Space Models offer:

- **Efficiency**: Linear-time complexity for long sequences
- **Flexibility**: Can be combined with other architectural components
- **Scalability**: Better memory usage than Transformers for long contexts
- **Competitive Performance**: Achieve results comparable to Transformers on many tasks

The field continues to evolve rapidly, with new architectures building on these foundations. As we move forward, the principles established by HiPPO, refined through the State-Space Model family, and perfected in Mamba, will likely influence the next generation of sequence modeling architectures.

---

*This lecture has traced the theoretical foundations and practical implementations that led from polynomial projections to modern selective state-space models, demonstrating how mathematical insights can drive architectural innovation in deep learning.*
