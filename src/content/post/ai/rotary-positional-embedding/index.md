---
title: "Introduction to Rotary Positional Embedding"
description: "This post introduces the concept of Rotary Positional Embedding and its implementation in transformer models."
publishDate: "12 April 2026"
updatedDate: "12 April 2026"
tags: ["AI", "TRANSFORMERS"]
---

## What is Positional Embedding?

Let's start with a simple sentence: "Big apple and small apple." In this sentence, we have two "apple". Even though they are the same word, they have different meanings. The first "apple" refers to a big apple, while the second "apple" refers to a small apple. To distinguish between these two "apple", we need to consider their positions in the sentence.(First "apple" is close to "big", while second "apple" is close to "small".)

![Positional Embedding example sentence](./positional-embedding-example-sentence.png)

## Previous approaches to positional embedding(Additive)

In the original transformer model, positional embedding is implemented as an additive method. Given a token embedding $x_t \in \mathbb{R}^d$, the positional embedding $PE(t) \in \mathbb{R}^d$ is added to it:

$$x_t + PE(t)$$

However, this additive method has a limitation. It cannot capture the relative position between tokens effectively. Adding the positional embedding and applying self-attention wouldn't model the relative position between tokens well.

## Rotary Positional Embedding(Multiplicative)

In the paper [ROFORMER: ENHANCED TRANSFORMER WITH ROTARY
POSITION EMBEDDING](https://arxiv.org/pdf/2104.09864), the authors propose a new method for positional embedding called Rotary Positional Embedding. This is done by matmul the token embedding with a rotation matrix.

## Terminology
$d$: Dimension of hidden state

$x_t, q_t, k_t$: Token embedding, Query, Key for $t$ th token. These are all in $\mathbb{R}^d$ unless specified.

$x_t', q_t', k_t'$: Rotated token embeddings, Queries, Keys for $t$ th token. These are all in $\mathbb{R}^d$ unless specified.

## Complex Space

### Representation of 2D vector in complex space

For any 2D vector $q_t = (q_{t0}, q_{t1}) \in \mathbb{R}^2$, we can represent it in the complex plane as:

$$q_t = |q| e^{i \alpha}$$

Where $|q|$ is the magnitude of the vector and $\alpha$ is the angle it makes with the positive x-axis. 

> This is all due to Euler's formula: $$e^{i \theta} = \cos(\theta) + i \sin(\theta)$$

![Complex plane](./complex-plane.png)

### Rotation in complex space

In the complex plane, we can perform a rotation by multiplying a complex number with another complex number that represents the rotation. For example, if we want to rotate a vector by an angle $\theta$, we can multiply it with $e^{i \theta}$:

$$
\begin{aligned}
q_t' = e^{i \theta} q_t
&= |q| e^{i (\alpha + \theta)}
\end{aligned}
$$

![Rotation in complex plane](./rotation-complex-plane.png)

## Details of Rotary Positional Embedding(with hidden_dim=2)

Let's consider a simple case where the hidden dimension is 2. 
Given a token embedding at position $t$, $x_t = (x_{t0}, x_{t1}) \in \mathbb{R}^2$, the rotary positional embedding is applied as a rotation in the complex plane.

First, let's define the token embeddings in complex plane: 

$$x_t = (x_{t0}, x_{t1}) = (|x| \cos(\alpha_t), |x| \sin(\alpha_t)) = |x| e^{i \alpha_t} $$

The rotation is defined as follows:
$$
\begin{aligned}
x_m' = e^{i \theta m} x_m = |x| e^{i (\alpha_m + \theta m)} \\
x_n' = e^{i \theta n} x_n = |x| e^{i (\alpha_n + \theta n)}
\end{aligned}
$$

$x_m$ rotates by $\theta m$ and $x_n$ rotates by $\theta n$. 

By doing so, if we apply inner product between $x_m'$ and $x_n'$, we can capture the relative position between $m$ and $n$:
> The inner product in complex space is defined as follows: $x_m' \cdot x_n' = Real(x_m' \overline{x_n'})$

$$
\begin{aligned}
x_m' \cdot x_n'
&= (|x| e^{i \theta  m}) \cdot (|x| e^{i \theta n}) \\
&= Real(|x| e^{i (\alpha_m + \theta m)} |x| e^{-i (\alpha_n + \theta n)}) \\
&= Real(|x|^2 e^{i (\alpha_m - \alpha_n + \theta m - \theta n)}) \\
&= |x|^2 \cos(\alpha_m - \alpha_n + \theta (m-n)) \ .. \ (Eq \ 1.)
\end{aligned}
$$

As you can see, the inner product between $x_m'$ and $x_n'$ depends on the relative position between $m$ and $n$ through the term $\theta (m-n)$. This allows the model to capture the relative positional information effectively.

> In real implementation, we don't use the concept of complex plane. Instead, we use 2D rotation matrix to achieve the same effect. In this post, I will not go into the details of 2D rotation matrix. If you are interested, check [here](https://krasserm.github.io/2022/12/13/rotary-position-embedding/)

## Expanding the RoPE into hidden_dim=$d$
In practice, the hidden dimension is usually much larger than 2. In this case, we can apply the same rotation to each pair of dimensions. For example, if the hidden dimension is 4, we can apply the same rotation to the first two dimensions and the last two dimensions:
$$
\begin{aligned}
x_m = (x_{m0}, x_{m1}, x_{m2}, x_{m3}) = ((x_{m0}, x_{m1}), (x_{m2}, x_{m3})) \\
x_n = (x_{n0}, x_{n1}, x_{n2}, x_{n3}) = ((x_{n0}, x_{n1}), (x_{n2}, x_{n3}))
\end{aligned}
$$

Assume $\theta_i = 10000^{-2(i-1)/d}$.

Applying rotation would be as follows:

1. Rotate the first two dimensions with $\theta_1$:
$$
\begin{aligned}
(x_{m0}', x_{m1}') = e^{i \theta_1 m} (x_{m0}, x_{m1}) \\
(x_{n0}', x_{n1}') = e^{i \theta_1 n} (x_{n0}, x_{n1})
\end{aligned}
$$
2. Rotate the last two dimensions with $\theta_2$:
$$
\begin{aligned}
(x_{m2}', x_{m3}') = e^{i \theta_2 m} (x_{m2}, x_{m3}) \\
(x_{n2}', x_{n3}') = e^{i \theta_2 n} (x_{n2}, x_{n3})
\end{aligned}
$$

> You may wonder why we use different $\theta$ for different dimensions. This is because we want to capture the positional information at different frequency. Recall ($Eq \ 1.$)