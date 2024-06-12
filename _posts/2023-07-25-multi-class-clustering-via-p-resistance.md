---
title: "Multi-class Graph Clustering via Approximated $p$-Resistance (Short Preview of Paper)"
date: 2023-07-25T09:34:30-00:00
show_date: true
categories:
 - blog
tags:
 - graph
 - ML
 - ICML2023
---

In this article, I'd like to preview our recent ICML paper.

Shota Saito and Mark Herbster. Multi-class Graph Clustering via Approximated Effective $p$-Resistance. In _Proc. ICML_
pp. 29697--29733, 2023


## To Exploit $p$ More for Multi-Class Clustering

We work towards multi-class clustering with better exploitation of "$p$." This is the motto.

For clustering problems, we often consider the cut over a graph defined as follows.
For a graph $G=(V,E)$, the cut $S_{G,2}$ is defined as

$$
S_{G,2}(\mathbf{x}) := \sum_{ij \in V} a_{ij} |x_{i} - x_{j}|^{2}.
$$

Sometimes, aiming for generalization of this cut, we consider the $p$-energy as

$$
S_{G,p}(\mathbf{x}) := \sum_{ij \in V} a_{ij} |x_{i} - x_{j}|^{p}.
$$

Note that these energies can be written as seminorm, which we call graph $p$-seminorm $$\|\mathbf{x}\|_{G,p}$$, as

$$
\|\mathbf{x}\|_{G,p}^{p} := S_{G.p}(\mathbf{x}).
$$

The most standard way to incorporate this $p$-energy is the spectral clustering of graph $p$-Laplacian.
However, the graph $p$-Laplacian has a major drawback for the multi-class clustering; obtaining higher eigenpairs of graph $p$-Laplacian is practically difficult.
The existing papers on $p$-Laplacian use a workaround for multi-class clustering.

This is difficult.
In a difficult situation, there are two ways we can choose; going forward or taking an alternative path.
Going forward is always a hero's move.
However, the difficulty of $p$-Laplacian is actually a paved way to hell; although the graph $p$-Laplacian has a broader research community, such as math and physics, and way longer history than ML, but still difficult.
Therefore, the hero's move actually leads to a solid unsolved problem, and the rational decision is to take an alternative path.


We want to exploit $p$ more for multi-class clustering, more than the existing workarounds on $p$-Laplacian. 
For this purpose, we consider an alternative; $p$-resistance.
The $p$-resistance is defined as

$$
r_{G,p}(i,j) := 1/(\min_{\mathbf{x}} \{S_{G,p} (\mathbf{x})\quad \mathrm{s.t}\ x_{i} - x_{j} = 1\}), \quad (\ast)
$$

The benefit of $p$-resistance is that $p$-resistance has a triangle inequality, such as

$$
r_{G,p}^{1/(p-1)}(i,j) \leq r_{G,p}^{1/(p-1)}(i,\ell) + r_{G,p}^{1/(p-1)}(\ell,j).
$$

Now, seeing these distance characteristics, some natural ideas for multi-class graph clustering have arisen.
We use $1/(p-1)$-th power of $p$-resistance as a distance, as we apply some distance-based clustering methods, such as $k$-center and $k$-medoids.

However, still, the problem remains.
It is computationally expensive to compute all the pairs of $p$-resistance since, for each pair, we need to solve the optimization problem from scratch.
To address this problem, we propose an approximation form of the $p$-resistance, as

$$
r_{G,p}(i,j) \approx \|L^{+}\mathbf{e}_{i} - L^{+}\mathbf{e}_{j}\|_{G,q}^{p}, \quad (\diamond)
$$

where $q$ satisfies $1/p + 1/q = 1$.
We show the quality bound of this approximation.
Also, if the graph is a tree, this approximation becomes exact, which indicates that this representation is promising.

Equation $(\diamond)$ aids to compute the effective resistance faster for many pairs.
With Equation $(\diamond)$, we compute $L^{+}$ first, and we obtain the effective resistance using Equation $(\diamond)$, and then we can recycle $L^{+}$.

Now, we have a way to compute $p$-resistance quickly.
The triangle inequality itself does not guarantee the clustering result.

By varying $p$, the $p$-resistance captures different topologies of the graph.
In the following example, we use the $k$-center algorithm on the distance obtained by $p$-resistance.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/presistance/illustrative.png" alt="">

Since the $p$-resistance corresponds to the shortest path when $p\to \infty$, we observe that the clustering result focuses more on path-based topology, that is, a preference for smaller shortest-path distances between vertices in the cluster.
On the contrary, since the $p$-resistance corresponds to the st-cut when $p\to 1$, the clustering result biases towards clusters with high internal connectivity, like a min-cut.
Using this characteristic, we expect varying $p$ tunes the clustering result somewhere suitable between cut-based and path-based.

We now also provide some theoretical justification for exploiting $p$-resistance for clustering.


## Collection of "Clips"

Now, I quickly wrapped up what is going on in our ICML paper, but the above is a preview. Here, I have a confession to make.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This is a promotional blog for my recent paper to ICML 2023. I&#39;ve almost never finished reading such promotional blogs because it&#39;s often too short to understand or too long for the blog format. Instead, I try to &quot;clip&quot; and highlight some parts of the paper at a reasonable length <a href="https://t.co/iEujeu4rxi">https://t.co/iEujeu4rxi</a></p>&mdash; Shota Saito (@ShotaSAITO) <a href="https://twitter.com/ShotaSAITO/status/1682009332479848451?ref_src=twsrc%5Etfw">July 20, 2023</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Yes, I've never got some ideas from this kind of "paper blog."
So, I try like a "clip" of a stream from Twitch or youtube at a reasonable length and to highlight some point that itself can stand as an interesting point.

I have written four of this kind of "clip" articles.

## 1. Limitaition of Graph $p$-Laplacian

[The first article][1] discusses why the higher order eigenpairs of graph $p$-Laplacian are difficult.

## 2. Effective Resistance and Graph

[The second article][2] explains the very introductory stuff; how effective resistance is formulated.

## 3. $p$-Resistance and its Approximation

[The third article][3] briefly explains what is the idea source of our recent paper approximation of $p$-resistance, how we obtain the guarantee, and briefly explains how $p$ works in the multi-class clustering.

## 4. Limitation of Effective Resistance and How $p$ Overcomes its Limitation

While people in graph ML heard effective resistance, I assume that many people, especially younger graph researchers, do not know the effective resistance. This might be because the effective resistance doesn't dance at the center; instead, spectral clustering and then GNN take the spot. [The fourth article][4] explains the limitation of resistance, which I speculate why the effective resistance has not been in the center and how $p$ overcomes the limitation.


***
Citation:

If you find this piece useful, please cite our paper.


```
@inproceedings{saito2023multi,
 title={Multi-class Graph Clustering via Approximated Effective $p$-Resistance.},
 author={Saito, Shota and Herbster, Mark},
 booktitle={Proc. ICML},
 pages = {29697--29733},
 year={2023}
}
```

[approx]: https://arxiv.org/abs/2306.08617
[1]: {{ site.url }}{{ site.baseurl }}/blog/limitation-of-graph-p-laplacian/
[2]: {{ site.url }}{{ site.baseurl }}/blog/effective-resistance/
[3]: {{ site.url }}{{ site.baseurl }}/blog/effective-p-resistance/
[4]: {{ site.url }}{{ site.baseurl }}/blog/limitation-of-resistance/