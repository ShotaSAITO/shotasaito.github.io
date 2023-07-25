---
title: "Graph to Effective $p$-resistance, its Approximation, and its Application to Multi-class Clustering"
date: 2023-07-21T15:34:30-00:00
show_date: true
categories:
  - blog
tags:
  - graph
  - ML
  - ICML2023
---

In this article, I'd like to explain some pieces of generalization of effective resistance to $p$-resistance and its approximation.

***

_Note: This article is a promotion of [our recent paper][approx] at ICML._

***

An analogy between graph and circuit offers us an effective resistance. 
Effective resistance serves as a distance over a graph, which is one of the reasons why effective resistance fascinates many researchers.
One may be curious here; what will happen if we generalize the energy from the square to $p$-th power?
In this article, I'd like to explain some foundations of $p$-resistance and our new results on $p$-resistance in the recent paper.


## Recap of Effective Resistance


To start, I'd briefly review the effective resistance. 
The following review is a short summary of [the previous post][resistancepost] on effective resistance.

The discussion below is built on the graph $p$-seminorm for a graph $G=(V,E)$, which is defined as

$$
\|\mathbf{x}\|_{G,p} := \|C\mathbf{x}\|_{\mathbf{w},p} = \left(\sum_{ij \in V} a_{ij} |x_{i} - x_{j}|^{p} \right)^{1/p}. \quad (\dagger)
$$

where $C \in \mathbb{R}^{|E| \times |V|}$ is an incidence matrix, $\mathbf{w} \in \mathbb{R}^{|E|}$ contains the graph weights, and $a_{ij}$ is a weight of the edge between the vertex $i$ and $j$. 
If the vector is under-scripted to the norm symbols, then the norm is a weighted norm by the vector. 
Note that the seminorm $$\|\cdot\|_{G,p}$$ is a seminorm since $$\|c\mathbf{1}\|_{G,p}=0$$.
Now, the rest of this section is about $p=2$ case.

We define the energy over a graph as

$$
S_{G,2} (\mathbf{x}) := \|\mathbf{x}\|_{G,2}^{2} = \sum_{i,j \in V} a_{ij} |x_{i} - x_{j}|^{2},
$$

and effective resistance between vertices $i$ and $j$ as

$$
r_{G,2}(i,j) := 1/(\min_{\mathbf{x}} \{S_{G,2} (\mathbf{x})\quad \mathrm{s.t}\ x_{i} - x_{j} = 1\}). (*)
$$

We know that this effective resistance serves as a distance. 
Notably, the effective resistance satisfies the triangle inequality, as

$$
r_{G,2} (i,j) \leq r_{G,2} (i,\ell) + r_{G,2} (\ell,j).
$$

The effective resistance can be analytically written as 

$$
r_{G,2} (i,j) = \|L^{+}\mathbf{e}_{i} - L^{+}\mathbf{e}_{j}\|_{G,2}^{2}, \quad (\dagger)
$$

where $L$ is an unnormalized graph Laplacian, and $L^{+}$ is a pseudoinverse of $L$.

Equation $(\dagger)$ aids to compute the effective resistance faster for many pairs.
Without Equation $(\dagger)$, for each pair, we need to solve each optimization problem of $(\ast)$ from scratch[^opt].
With Equation $(\dagger)$, we compute $L^{+}$ first, and we obtain the effective resistance using Equation $(\dagger)$, and then we can recycle $L^{+}$.


## $p$-Resistance and Triangle Inequality

Like the generalization from the graph Laplacian to $p$-Laplacian, it is one of the universe principles that people want to expand to $p$-seminorm (See [my previous post][plaplacianost] on this).
The effective $p$-resistance is defined as

$$
r_{G,p}(i,j) := 1/(\min_{\mathbf{x}} \{S_{G,p} (\mathbf{x})\quad \mathrm{s.t}\ x_{i} - x_{j} = 1\}), \quad (**)
$$

where

$$
S_{G,p} := \|\mathbf{x}\|_{G,p}^{p} = \sum_{i,j \in V}a_{ij} |x_{i} - x_{j}|^{p}.
$$

Like a graph $p$-Laplacian, the $p$-resistance improves the performance in various areas of machine learning, such as semi-supervised learning and online learning.
Tuning $p$ of $p$-resistance captures different topological characteristics of the graph, which I will discuss later.

We now see the benefit of the $p$-resistance, improving some ML problems' performance.
However, unfortunately, this $p$-resistance does not have a triangle inequality.
I hear your sigh. 
Don't get depressed.
I won't let you down.
Instead, $r_{G,p}^{1/(p-1)}(i,j)$ has a triangle inequality, i.e.,

$$
r_{G,p}^{1/(p-1)}(i,j) \leq r_{G,p}^{1/(p-1)}(i,\ell) + r_{G,p}^{1/(p-1)}(\ell,j).
$$

See [Herbster][triangle] (2010) for the details.

With the triangle inequality, you might feel some potential of the $p$-resistance.
However, when we want this $p$-resistance for many pairs, do we need to solve the optimization problem for many pairs?
Do we have a similar representation Equation $(\dagger)$ for $p=2$ case, which reduces the computational time?
The natural idea in your head might be as follows.
Looking at Equation $(\dagger)$, does there exist some norm $$\|\cdot\|^{\star}$$ such that $$r_{G,p}(i,j) = \|L^{+}\mathbf{e}_{i} - L^{+}\mathbf{e}_{j}\|^{\star}$$.
If this were the case, the $p$-resistance would be beautiful.

Straight to the point. We haven't seen such a beautiful thing yet for a general graph, unfortunately. 
But, now, in [our recent paper][approx], this $p$-resistance is well approximated by 

$$
r_{G,p}(i,j) \approx \|L^{+}\mathbf{e}_{i} - L^{+}\mathbf{e}_{j}\|_{G,q}^{p}, \quad (\diamond)
$$

where $q$ satisfies $1/p + 1/q = 1$.
How good is this approximation? 
We now have a quality bound in our paper.
The approximation rate can be shown using Hölder's inequality, which is similar to the fact that we can show Equation $(\ast)$ using the Cauchy-Schwarz inequality.
Also, if the graph is a tree, this approximation becomes exact, which indicates that this representation is promising.

Using $(\diamond)$, for the same reasons as when $p=2$, we can compute this approximated $p$-resistance much faster than solving the optimization problem for many pairs.

## $p$-Resistance and Clustering: how $p$ captures topology of graph

One application of $p$-resistance is clustering.
As I lightly mentioned above, the $p$ of $p$-resistance captures a topology of the graph as follows;

- When $p \to 1$, the $p$-resistance corresponds to st-min cut.
- When $p = 2$, this is the standard effective resistance.
- When $p \to \infty$, the $p$-resistance corresponds to the shortest path.

See [Alamgir and von Luxburg](phase) (2011, NIPS) for the details.


The following are illustrative examples of using $p$-resistance for the clustering.
In the following example, we use the $k$-center algorithm on the distance obtained by $p$-resistance.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/presistance/illustrative.png" alt="">

Since the $p$-resistance corresponds to the shortest path when $p\to \infty$, we observe that the clustering result focuses more on path-based topology, that is, a preference for smaller shortest-path distances between vertices in the cluster.
On the contrary, since the $p$-resistance corresponds to the st-cut when $p\to 1$, the clustering result biases towards clusters with high internal connectivity, like a min-cut.
Using this characteristic, we expect varying $p$ tunes the clustering result somewhere suitable between cut-based and path-based.

## Conclusion

In this article, I introduced $p$-resistance, a generalization of the graph effective resistance.
Moreover, I highlight the result from our recent paper, the approximation of $p$-resistance.
I also explained how $p$ captures the topology of the graph and applies it to the clustering.

If you come up with some application of all the pairs of the $p$-resistance, just let me know!

***
Advert:

The most popular manner in ML to exploit graph $p$-seminorm is the graph $p$-Laplacian.
However, we know that it is difficult to bridge between the theory and practice of multi-class clustering using graph $p$-Laplacian (for details, see [the previous post][plaplacianpost]). 
The 1/(p-1)-th power of $p$-resistance seems promising for multi-class graph clustering.
Can we have theoretical support for using $p$-resistance for multi-class clustering? 
Yes, we can.
If you are interested, please see [our paper][approx].

***
Citation:

If you find this piece useful, please cite this paper.

```
@inproceedings{saito2023multi,
 title={Multi-class Graph Clustering via Approximated Effective $p$-Resistance.},
 author={Saito, Shota and Herbster, Mark},
 booktitle={Proc. ICML},
 pages = {29697--29733},
 year={2023}
}
```

## Small note on the variants of the $p$-resistance

In the ML field, two variants of $p$-resistance exist, with slight differences.

[Lever and Herbster][seminorm] (2009, COLT) discussed the $p$-resistance induced from the energy defined as

$$
\sum_{i,j \in V} a_{ij} |x_{i} - x_{j}|^{p}.
$$

[Alamgir and von Luxburg][phase] (2011, NIPS) explored the characteristics of $p$-resistance induced form the energy defined as

$$
\sum_{i,j \in V} a^{q-1}_{ij} |x_{i} - x_{j}|^{q}.
$$

Essentially, these two are not that different, i.e., they share the same characteristics.
One thing to note is that the $p$ of Lever and Herbster's $p$-resistance and $p$ of the Alamgir and von Luxburg are opposite; when one says $p \to 1$, the other says $p \to \infty$ and vice versa.

The energy by Lever and Herbster is more compatible with the other graph $ p$-seminorm-based machine learning, such as the graph $p$-Laplacian.
The energy by Alamgir and von Luxburg is more compatible with literature from circuit theory. Even an old textbook has such a definition, e.g., page 176 of [Soardi][soardi] (1994).



[^opt]: Even though this constrained optimization problem can be rewritten less computationally expensive unconstrained optimization problem, it is still expensive to compute.

[plaplacianpost]: {{ site.url }}{{ site.baseurl }}/blog/limitation-of-graph-p-laplacian/
[resistancepost]: {{ site.url }}{{ site.baseurl }}/blog/effective-resistance/
[triangle]: https://discovery.ucl.ac.uk/id/eprint/1311162/
[approx]: https://arxiv.org/abs/2306.08617
[phase]: https://papers.nips.cc/paper_files/paper/2011/hash/07cdfd23373b17c6b337251c22b7ea57-Abstract.html
[seminorm]: https://www.learningtheory.org/colt2009/papers/016.pdf
[soardi]: https://link.springer.com/book/10.1007/BFb0073995