---
title: "Limitation of Effective Resistance in ML and How $p$-Resistance Overcomes the Limitations"
date: 2023-07-22T09:34:30-00:00
show_date: true
categories:
 - blog
Tags:
 - graph
 - ML
 - ICML2023
---

In this article, I'd like to explain the limitations of effective resistance in the ML area and how its generalization to $p$ overcomes this problem.

***

_Note: This article is a promotion of [our recent paper][approx] at ICML._

***

## TL; DR
The 2-resistance has some limitation, which is overcome by generalizing from 2-resistance to $p$-resistance.

## The Opener

Effective resistance is a mathematically fun tool.
Due to its fun properties, effective resistance has fascinated people.
At the same time, the effective resistance is slightly away from the graph machine learning limelight, while as time goes by, spectral clustering and then GNN enjoy the spot.

This may have a reason. 
Below I'll explain this, but before that, I'd want to establish the notations briefly.

## Recap of Effective Resistance

Let $G=(V,E)$ be a graph.
The effective resistance over a graph is considered from an analogy between a graph and a circuit.
We define the energy over a graph as

$$
S_{G,2} (\mathbf{x}) := \sum_{i,j \in V} a_{ij} |x_{i} - x_{j}|^2,
$$

where $\mathbf{x}$ is a voltage distribution.
The effective resistance between vertices $i$ and $j$ defined as

$$
r_{G,2}(i,j) := 1/(\min_{\mathbf{x}} \{S_{G,2} (\mathbf{x})\quad \mathrm{s.t}\ x_{i} - x_{j} = 1\}).
$$

Note that the effective resistance can be thought of as the inverse of the two-pole semi-supervised learning (SSL) problem, i.e., the SSL problem with two fixed labels.

This effective resistance is generalized to $p$-resistance as follows.
We consider energy as

$$
S_{G,p} (\mathbf{x}) := \sum_{i,j \in V} a_{ij} |x_{i} - x_{j}|^p,
$$

and the effective $p$-resistance as

$$
r_{G,p}(i,j) := 1/(\min_{\mathbf{x}} \{S_{G,p} (\mathbf{x})\quad \mathrm{s.t}\ x_{i} - x_{j} = 1\}).
$$

The above is a dry collection of definitions of effective resistance, and all the funs are eliminated.
I refer to my previous posts (post on effective resistance and effective $p$-resistance) for fun details.

One wet thing is that we see the connection between SSL and effective resistance. 

Let us consider the following SSL problem.

$$
\mathbf{x}^{ij,p*}:=\text{argmin}_{\mathbf{x}} S_{G,p} (\mathbf{x}) \quad \text{s.t. } x_{i} - x_{j} = 1 \quad (\dagger)
$$

Then, for the $p=2$ case, the solution of this SSL $\mathbf{x}^{ij,2*}$ and the effective resistance has a connection as

$$
 x^{ij,2*}_{i} - x^{ij,2*}_{\ell} \geq x^{ij,2*}_{\ell} - x^{ij,2*}_{j} \iff r_{G,2} (i, \ell) \geq r_{G,2} (\ell,j) \quad (\ast)
$$

for any third point $\ell$.
This shows the equivalence of the effective resistance and the solution of the SSL problem.
This is helpful, but it's been an open problem if this holds for the general $p>1$.
See the [paper][alamgir] by Alamgir et al. (2010, NIPS) for the details. 

Spoiler Alert: [Our new paper][approx] resolves this open problem positively.


## Limitation of Effective Resistance

Let's get back to where we were.
What I wanted to discuss is why this effective resistance has not been hitting the limelight.

Below we consider the scenario where the graph is constructed from vector data by some method, such as $\epsilon$-graph, $k$-NN graph, and Gaussian kernel.

I thought this might have a reason.
The reason I speculate is that several papers pointed out the limitations. 
Consider constructing a graph from vector data, such as Gaussian kernel or $\epsilon$-graph. 
Under this assumption, the major papers point out this limitation, e.g., 

- Nadler et al. [Semi-Supervised Learning with the Graph Laplacian: The Limit of Infinite Unlabelled Data][nadler] In Proc. NIPS, 2009.
- von Luxburg et al. [Getting lost in space: Large sample analysis of the resistance distance][vonluxburg] In Proc. NIPS, 2010

What sensational headlines from a tabloid called NIPS[^NIPS].
This sounds scandalous for effective resistance.
Although the effective resistance is a charming folk, this folk might have resigned from the position if the effective resistance were a politician.

I made a joke here, but the contents of these papers are solid indeed.
The effective resistance is reported to converge to the meaningless function if you draw an infinite number of the data points, i.e., in the asymptotic condition, 

$$
r_{G,2} (i,j) \to \frac{1}{d_{i}} + \frac{1}{d_{j}},
$$

where $d_{j} := \sum_{i \in V} a_{ij}$ is a degree of the vertex $j$.

We probably thought that the effective resistance is a more cool folk, i.e., reflecting a more nuanced graph structure.
However, the effective resistance does not reflect the whole graph structure.
Instead, under some circumstances, this only considers partial structures, the number of vertices connected to $i$ and $j$. 
This sounds less sexy.


Also, Nadler et al. showed that under some similar assumptions to von Luxburg, the solution of the SSL problem converges to constant except for the near constraints if we draw an infinite number of data points.
Therefore, since the solution is almost constant, no learning occurs in the SSL problem $(\dagger)$.

It is not hard to expect that the solution to the SSL problem and the 2-resistance is related since the denominator of the 2-resistance is actually the SSL problem.
However, it is not trivial *how* these are related.
Now, the relationship Equation $(\ast)$ shows that Nadler et al. and von Luxburg corroborate each other. If one of the SSL or effective resistance is meaningless, the other is also meaningless.

To sum up this section, the effective resistance is less sexy than we thought. 
Less sexy is my word, but the papers describe it as *meaningless* or *having a limit*.
This is one of the reasons **I'm speculating** why the effective resistance doesn't enjoy the spot.


## Here is a hero, $p$

How do we overcome this problem?
The generalization from 2-resistance to $p$-resistance will help.
In short, larger $p$ overcomes this problem.

[Alamgir et al.][alamgir] (2010, NIPS) shows that under some assumptions, the $p$-resistance converges to 

$$
r_{G,p}(i,j) \to \frac{1}{d^{p-1}_{i}} + \frac{1}{d^{p-1}_{j}},
$$

that is also a meaningless function when $p$ is small.
However, when $p$ is large, the $p$-resistance converges to a value associated with a more nuanced graph structure.

Furthermore, the solution of the semi-supervised learning for general $p>1$ converges to a non-meaningless function when $p$ is large; [Alaoui et al.][Alaoui] (2016, JMLR) showed the case of $p$ is an integer, and [Slepcev et al.][Slepcev] (2019, SIMA) generalized the result to the general $p>1$.
For small $p$ again, the solution converges to a meaningless function.


Again, those two seem related. 
Do we have some relationship like Equation $(\ast)$? 
It's been an open problem if this holds for any $p>1$ or not. 
Now, [our recent paper][approx] resolved this positively, i.e.,

$$
 x^{ij,p*}_{i} - x^{ij,p*}_{\ell} \geq x^{ij,p*}_{\ell} - x^{ij,p*}_{j} \iff r_{G,p} (i, \ell) \geq r_{G,p} (\ell,j) \quad (\ddagger).
$$

The relationship $(\ddagger)$ shows that Alamgir et al. and Slepcev corroborate each other.
If one of the SSL or effective resistance is meaningless, the other is also meaningless when $p$ is small, while when $p$ is large, if the one is meaningful, the other is also meaningful.

## Hero Isn't Perfect



## Conclusion


While the effective resistance is fun, the resistance enjoys the limelight less in the ML area.
This might be due to some limit in the asymptotic condition. 
This article promotes that if we generalize from 2-resistance to $p$-resistance, this limitation is overcome when $p$ is large.

*** 
Note:

The papers discussing the SSL in this blog post showed the results not only for the two known label cases but also for any number of known labels.
Also, since the assumptions vary from paper to paper, if you reuse the results, please check each paper before you use it.
The relationships $(\ast)$ and $(\ddagger)$ hold for any graph that is not necessarily made from vector data. 

***
Advert:

We use the relationship $(\ddagger)$ to justify the use of $p$-resistance as a distance for the multi-class clustering.

If you are interested in one of the applications of resistance or $p$-resistance, please see Sec. 4.1 in [our paper][approx].


***
Citation:

If you find this piece useful, please cite our paper.
Appendix A provides a comprehensive review of the topic of this article, the limitation of the $p$-Laplacian for multi-class clustering.

```
@inproceedings{saito2023multi,
 title={Multi-class Graph Clustering via Approximated Effective $p$-Resistance.},
 author={Saito, Shota and Herbster, Mark},
 booktitle={Proc. ICML},
 pages = {29697--29733},
 year={2023}
}
```

[^NIPS]: Just to follow up. This is just a joke. 
[approx]: https://arxiv.org/abs/2306.08617
[nadler]: https://proceedings.neurips.cc/paper_files/paper/2009/hash/68ce199ec2c5517597ce0a4d89620f55-Abstract.html
[vonluxburg]: https://proceedings.neurips.cc/paper_files/paper/2010/hash/0d0871f0806eae32d30983b62252da50-Abstract.html
[alamgir]: https://papers.nips.cc/paper/2011/hash/07cdfd23373b17c6b337251c22b7ea57-Abstract.html
[Alaoui]: http://proceedings.mlr.press/v49/elalaoui16.pdf
[Slepcev]: https://arxiv.org/pdf/1707.06213