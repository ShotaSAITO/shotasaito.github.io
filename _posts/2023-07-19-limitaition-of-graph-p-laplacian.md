---
title: "Limitation of graph $p$-Laplacian for multi-class graph clustering"
date: 2023-07-19T15:34:30-00:00
show_date: true
categories:
  - blog
tags:
  - graph
  - ML
  - ICML2023
---

In this article, I'd like to explain some practical limitations of the graph $p$-Laplacian for multi-class graph clustering.

***

_Note: This article is a promotion of [our recent paper][approx] at ICML._

***


People want multi-class clustering for a graph. 
If you don't agree with it, I think the rest of the article is entirely nonsense.

For graph clustering, spectral clustering is the most popular method.
The spectral clustering uses the smallest $k$ eigenvectors of the graph Laplacian $L$ for $k$-class clustering.
In this article, we assume that the graph Laplacian is unnormalized, but the same discussion can also apply to normalized Laplacian.


The use of the $k$-smallest eigenvectors for clustering is theoretically supported.
There are several ways to justify this, which indicates that spectral clustering is a good theory.
One way to justify this is the Cheeger inequality. 
The Cheeger inequality gives a quality assurance for the results of the multi-class clustering using the eigenvectors of $L$.

Behind the scene of this spectral clustering, graph 2-seminorm $$\|\mathbf{x}\|_{G,2}$$ for a graph 
$G=(V,E)$ and a vector $\mathbf{x} \in \mathbb{R}^{|V|}$ and plays a role, defined as

$$
\|\mathbf{x}\|_{G,2}^{2} := \|C\mathbf{x}\|_{\mathbf{w},2}^{2} = \sum_{ij\in V} a_{ij}|x_{i} - x_{j}|^{2}, \quad (*)
$$

where $C \in \mathbb{R}^{|E| \times |V|}$ is an incidence matrix, $\mathbf{w} \in \mathbb{R}^{|E|}$ contains the graph weights, and $a_{ij}$ is a weight of the edge between the vertex $i$ and $j$. 
If the vector is under-scripted to the norm symbols, then the norm is a weighted norm by the vector. 
If the matrix is under-scripted to the norm symbols, then the norm is induced by that matrix.
Sorry for the sudden many symbols, don't freak out with these notations. 

I note that

$$
\|\mathbf{x}\|_{G,2}^{2} = \|C\mathbf{x}\|_{\mathbf{w},2}^{2} = \|\mathbf{x}\|_{L}^{2} = \mathbf{x}^{\top}L\mathbf{x}, \quad (\circ)
$$

due to the basic fact that $L=C^{\top}\mathrm{diag}(\mathbf{w})C$.

By the way, the graph seminorm is a seminorm because $ || c\mathbf{1} ||_{G,2} = 0$.
In this article, I say seminorm, but if you are a bit confused about the word seminorm, you can simply read seminorm as norm.

I suppose that the first two objects in the equations (*) are not that familiar to non-spectral people. 
But the last two, the quadratic form using the graph Laplacian and its expansion in Equation $(\circ)$, are poplar, known as the graph cut.
The $k$-th eigenpair $(\lambda_{k}, \mathbf{x}_{k})$ can be obtained through the following sequence using the graph 2-seminorm as

$$
\mathbf{x}_{k} = \mathrm{argmin}_{\mathbf{x}} \frac{\|\mathbf{x}\|_{G,2}^{2}}{\|\mathbf{x}\|_{2}^{2}} \quad \mathrm{s.t.}\ \mathbf{x}_{k} \bot \mathbf{x}_{1},\ldots,\mathbf{x}_{k-1}, 
$$

which can be rewritten in a more abstract way as

$$
\lambda_{k} = \min_{U| \mathrm{dim}(U)=k}\max_{\mathbf{x} \in U}\frac{\|\mathbf{x}\|_{G,2}^{2}}{\|\mathbf{x}\|_{2}^{2}}  \quad(\dagger).
$$

This sequence is known as min-max theorem or variational theorem.
This is how the graph 2-seminorm connects from the graph cut to the graph Laplacian and even to the spectral clustering.


Okay, so far, we saw a very brief connection between the 2-seminorm and the spectral clustering.
If people see the 2-seminorm, it is one of the universe principles that people want to expand to $p$-seminorm.
Here, I define the graph $p$-seminorm as

$$
\|\mathbf{x}\|_{G,p} := \|C\mathbf{x}\|_{\mathbf{w},p} = \left(\sum_{ij \in V} a_{ij} |x_{i} - x_{j}|^{p} \right)^{1/p}. \quad (\ddagger)
$$

By varying this $p$, the community observes "performance improvement" in the areas like clustering, semi-supervised learning, online learning, and more.

Our focus is clustering. 
Seeing the Equation (*), now the graph $p$-Laplacian is defined $\Delta_{p}$ as

$$
(\Delta_{p}\mathbf{x})_{i} := \sum_{j \in V} a_{ij} |x_{i}-x_{j}|^{p-1} \mathrm{sgn}(x_{i}-x_{j}).
$$

Observe that when $p=2$, this definition corresponds to the graph Laplacian as we already know as the matrix $L$.
The eigenpair of the graph $p$-Laplacian $(\lambda, \mathbf{x})$ is defined to satisfy the following relationship as

$$
(\Delta_{p}\mathbf{x})_{i} = \lambda|x_{i}|^{p-1}\mathrm{sgn}(x_{i}).
$$

The higher-order eigenvectors of the graph $p$-Laplacian have theoretical support for the use of spectral clustering; the Cheeger inequality is there for you.
We have a very similar thing for $p$-Laplacian to the standard Cheeger inequality.
I do not go into the details because this needs complicated setups but see [Tudisco and Hein][nodal] (2018) for more details.


So far, the multi-class clustering using $p$-Laplacian is shiny.
We have an exciting math to toy with.
We enjoy nice theoretical properties such as Cheeger inequality.
But this shine does not last long. 
Here come clouds.

The most severe limitation of the graph $p$-Laplacian is that we do not know how to obtain the higher-order eigenvectors.
In the following I explain the situation.
We know the second eigenvector of the graph $p$-Laplacian is given as

$$
\mathrm{argmin}_{\mathbf{x}}\frac{\|\mathbf{x}\|_{G,p}^{p}}{\min_{\eta}\|\mathbf{x} - \eta \mathbf{1}\|_{p}^{p}}.
$$

However, no one on earth knows this kind of exact identification of the third or higher.

We do not have an exact identification. I understand that. But do we have a similar sequence to Equation $(\dagger)$?
Yes. Yes? What a relief.
The following gives such a sequence, but spoiler alert, the sequence is very abstract.

The generalized dimension called _Krasnoselskii genus_ $\gamma$ is defined as

$$
 \gamma(B) =
 \left\{
 \begin{array}{l}
 0 \mathrm{\ if}\ B=\emptyset \\
 \inf \{k \in \mathbf{Z}^{+} \mid \exists \mathrm{odd\ continuous}\ h:B \rightarrow \mathbf{R}^{k}\backslash \{0\} \} \\
 \infty \mathrm{\ when\ no\ such\ }h \mathrm{\ exists\ } \forall j \in \mathbf{Z}^{+}
 \end{array}
 \right.
$$

Consider the set $$\mathcal{F}_k:= \{ B \mid B = -B, \mathrm{\ closed\ }, \gamma(B) \geq k \}$$. 
The sequence is defined as

$$
\lambda_{k} = \min_{B \subset \mathcal{F}_k} \max_{\mathbf{x} \in B} \frac{\|\mathbf{x}\|_{G,p}^{p}}{\|\mathbf{x}\|_{p}^{p}} 
$$

gives the eigenvalue of graph $p$-Laplacian, and the corresponding $\mathbf{x}$ gives the eigenvector.

Ooooooookay. We have a sequence. Yes, we do. Yes, you do.
But, this is a huge but, the limitation is that no one knows how to _implement_ (i.e., teach to computers) this Krasnoselskii genus at this point due to its abstract nature.
Remark that we do not know if this sequence exhausts all the spectra of the graph $p$-Laplacian[^cheeger].

Now, we see the limitation of using the graph $p$-Laplacian for multi-class clustering.
To summarize, we do not know the exact identification for the higher eigenpairs, and we do not know how to implement the sequence.
Thus, we can say that although we have theoretical support for using the higher eigenvectors of graph $p$-Laplacian, we do not know how to obtain them.
The existing research pieces use the workaround for this limitation. 


If you are interested in this content, please refer to the Appendix A of [our paper][approx], which gives a comprehensive review of this limitation.

***
Advert:

We see the limitation of the graph $p$-Laplacian.
Now, the question is, how can we exploit graph $p$-seminorm more?
We know varying $p$ improves the performance of the various problems.
However, the $p$-Laplacian does not enjoy the full of the potential that $p$-seminorm can have, since we have a limitation as we saw above.
Therefore, if we find some way that enjoys more graph $p$-seminorm, this can improve the performance of multi-class clustering.
Yes, our paper proposes the way.


***
Citation:

If you find this piece useful, please cite our paper.
Appendix A provides a comprehensive review for the topic of this article, limitation of the $p$-Laplacian for multi-class clustering.

```
@inproceedings{saito2023multi,
  title={Multi-class Graph Clustering via Approximated Effective $p$-Resistance.},
  author={Saito, Shota and Herbster, Mark},
  booktitle={Proc. ICML},
  pages = {29697--29733},
  year={2023}
}
```

[^cheeger]: The Cheeger inequality for the graph $p$-Laplacian is only for the eigenpairs obtained by the sequence above.
[approx]: https://arxiv.org/abs/2306.08617
[nodal]: https://arxiv.org/abs/1602.05567