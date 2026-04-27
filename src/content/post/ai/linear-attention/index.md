---
title: "Mathematical Formulation of Linear Attention"
description: "This article explores the mathematical formulation of linear attention, a technique used in machine learning to improve softmax attention mechanisms. We will discuss how linear attention works, its advantages, and its applications in various AI models."
publishDate: "23 April 2026"
updatedDate: "23 April 2026"
tags: ["AI"]
---

## Summary
Linear attention is a technique that modifies the traditional softmax attention mechanism to reduce computational complexity while maintaining fixed memory usage. This article explores the mathematical formulation of linear attention, its advantages over softmax attention.

## Terminology

- $q_t \in \mathbb{R}^{1 \times d}$: the query vector at time $t$
- $k_t \in \mathbb{R}^{1 \times d}$: the key vector at time $t$
- $v_t \in \mathbb{R}^{1 \times d}$: the value vector at time $t$
- $o_t \in \mathbb{R}^{1 \times d}$: the output vector at time $t$
- $Q \in \mathbb{R}^{L \times d}$: the matrix of all query vectors
- $K \in \mathbb{R}^{L \times d}$: the matrix of all key vectors
- $V \in \mathbb{R}^{L \times d}$: the matrix of all value vectors
- $O \in \mathbb{R}^{L \times d}$: the matrix of all output vectors
- $L$: the sequence length
- $d$: the dimension of the query, key, and value vectors
- $M \in \mathbb{R}^{L \times L}$: the causal mask matrix, where $M_{ij} = 0$ if $i \geq j$ and $\text{-inf}$ otherwise
- $S_{t} \in \mathbb{R}^{d \times d}$: the state matrix at $t$th token, which is the sum of outer products of key and value vectors up to time $t$: $S_{t} = \sum_{i=1}^{t} k_i^T v_i$
- $G_t \in \mathbb{R}^{d \times d}$: the forgetting gate matrix at $t$th token, which controls the contribution of each token to the state matrix
- $\odot$: element-wise multiplication operator

## Recap of Softmax Attention
Let's first recall the formulation of softmax attention. 

Parallel form:
$$
O = \text{softmax}\left(QK^T + M\right)V
$$

Recurrent form:
$$
O_t = \sum_{j=1}^t \frac{\text{exp}\left(q_tk_j^T\right)}{\sum_{l=1}^t \text{exp}\left(q_tk_l^T\right)} v_j
$$

The time complexity of softmax attention for prefill is $\text{O}\left(L^2d\right)$.

> You may wonder why. Let's think step-by-step by looking at parallel form equation.
> 1. $QK^T$: Matmul takes $\text{O}\left(L^2d\right)$ time.
> 2. $QK^T + M$: Summing the mask takes $\text{O}\left(L^2\right)$ time.
> 3. $\text{softmax}\left(QK^T + M\right)$: Softmax takes $\text{O}\left(L^2\right)$ time.
> 4. $\text{softmax}\left(QK^T + M\right)V$: Matmul takes $\text{O}\left(L^2d\right)$ time.
> Therefore, the overall time complexity is dominated by the $\text{O}\left(L^2d\right)$ terms, resulting in $\text{O}\left(L^2d\right)$ time complexity for softmax attention.


## Formulation of Linear Attention
Linear attention modifies the softmax attention by removing the softmax function. The formulation of linear attention is as follows:

Parallel form:
$$
O = (QK^T \odot \text{Mask})V
$$

where $\text{Mask} \in \mathbb{R}^{L \times L}$ is the lower triangular causal mask with $\text{Mask}_{ij} = 1$ if $i \geq j$ and $0$ otherwise.

Recurrent form:
$$
O_t = \sum_{j=1}^t \left(q_tk_j^T\right)v_j = \sum_{j=1}^t q_t\left(k_j^Tv_j\right) = q_t \sum_{j=1}^t k_j^Tv_j
$$

The time complexity of linear attention for prefill is $\text{O}\left(Ld^2\right)$.

> You may wonder why. Let's think step-by-step by looking at recurrent form equation.
> 1. $k_j^Tv_j$: Matmul takes $\text{O}\left(d^2\right)$ time.
> 2. $\sum_{j=1}^L k_j^Tv_j$: Summing over $L$ tokens takes $\text{O}\left(Ld^2\right)$ time.

## Expanding Linear Attention with State Matrix
We can further optimize linear attention by introducing a state matrix $S_t$ that accumulates the outer products of key and value vectors up to time $t$. The formulation becomes:
$$
\begin{aligned}
O_t &= \sum_{j=1}^t \left(q_tk_j^T\right)v_j \\
&= \sum_{j=1}^t q_t\left(k_j^Tv_j\right) \\
&= q_t \sum_{j=1}^t k_j^Tv_j \\
&=q_t S_t
\end{aligned}
$$

where $S_t = \sum_{j=1}^t k_j^T v_j$.

Each time we receive a new token, we can update the state matrix $S_t$ by adding the outer product of the new key and value vectors:
$$
S_t = S_{t-1} + k_t^T v_t
$$

## Limitation of Linear Attention
While linear attention reduces the computational complexity, it has a limitation in terms of expressiveness. The context is stored in a single state matrix $S_t \in \mathbb{R}^{d \times d}$. This means that the model can only capture interactions between keys and values in a limited way, as all information is compressed into a single matrix. 

> State matrix $S_t$ can be viewed as a reconstructing function that reconstructs the value vector $v_j$ from the key vector $k_j$. 

Let's see why. Let's try to reconstruct $v_l$ from $S_t$:
Assume $k_i$'s are normalized to unit length.
$$
\begin{aligned}
k_l^T S_t &= \sum_{j=1}^t k_l k_j^T v_j \\
&= v_l + \sum_{j \ne l} k_l k_j^T v_j
\end{aligned}
$$

As you can see from the equation, we can reconstruct $v_l$ from $S_t$ by multiplying $S_t$ with $k_l^T$. However, there is a noise term $\sum_{j \ne l} k_l k_j^T v_j$ that comes from other tokens. This noise term can make it difficult for the model to accurately capture the interactions between keys and values, especially when the sequence length is long.

## How modern Linear Attention Are Evolving
To address the limitation of linear attention, modern linear attention mechanisms such as Mamba2, GLA use decaying(forgetting) mechanism to reduce the noise term. By applying a decay factor to the state matrix $S_t$, the model can give more weight to recent tokens and less weight to older tokens, which helps to mitigate the noise from distant tokens.

$$
\begin{aligned}
S_t = G_t \odot S_{t-1} + k_t^T v_t
\end{aligned}
$$

$G_t$ differs per mechanism.

## References
- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Katharopoulos et al. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). ICML 2020.
- Schlag, Irie, Schmidhuber. [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174). ICML 2021.               
- Yang et al. [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635). ICML 2024.   
- Gu & Dao. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752). 2023.           
- Dao & Gu. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). ICML 2024.
- Songlin Yang. [DeltaNet Explained](https://sustcsonglin.github.io/blog/2024/deltanet-1/) — Part I (covers linear attention foundations, state matrix interpretation)