---
layout: post
title: "Linear Algebra for Machine Learning"
date: "2017-06-05"
status: [Under construction]
mathjax: true
published: false
categories: [Math, Machine Learning, Linear Algebra]
---

### Key Concepts
In Machine Learning and statistics, we are interested in modelling the relationships between our data. It is often useful to think about data points as being points in high dimensional space. In ML, we are interested in fitting a decision surface which best separates the datapoints by optimizing some loss function. This can either be done iteratively through [Calculus]({{ site.baseurl}}{% link _posts/2018-04-01-Calculus-for-ML.md %}), or through Linear Algebra techniques. Linear algebra is 
* a set of tools for describing and manipulating these high-dimensional objects, for the purpose of solving general systems of linear equations.
* a mathematical abstraction which studies the linear maps between vector spaces.

**Data Representation, Scalars, Vectors, Matrices, Tensors**

<u>Data Representation</u>
Data are typically represented in the form of scalars, vectors, matrices and Tensors. A single datapoint $x$, can be represented by a vector of $D$ scalar numbers, where each scalar represents a particular feature of the data point. The datapoint, $x \in \mathbb{R}^D$ can be written as $x=<x_1, x_2, ... x_D>$ represents an element that lives in our $D$-dimensional vector space.

<u>Vectors</u>
Geometrically, vectors describe the magnitude and direction of a point away from the origin in the $D$-dimensional Euclidean space. Each index in the vector gives the coordinate positions along that axis. Algebraic operations such as addition, subtraction, multiplication, division can thus be seen as changing the point in space by changing the magnitude and direction of the original vector.

<u>Matrices</u>

Matrices are a 2-D array of numbers, which has many different use cases and is highly versatile for representing data in Machine Learning. A matrix with $m$ rows and $n$ columns, could be
 
* Datapoints arranged in $m$ rows, and $n$ columns, where $m$ is the number of datapoints and $n$ is the number of features. 
* A 2-D greyscale image of $n$ height and $m$ width, where each element in the matrix is a pixel in the image.
* Weights and derivatives (the Jacobian matrix) for the activation layer of a neural network.
* $n$ Authors and $m$ objects in recommender systems.
* An undirected graph of $n$ verticies, which can be modelled as a matrix with the $i, j$ entry containing the number of edges from vertex $i$ to vertex $j$.

<u>Matrix Multiplication</u>

The matrix multiplication, or matrix product of two matrices $A^{n\times m}$ and $B^{m\times p}$ gives a third matrix $C^{n\times p}$. For each element in $C=AB$, we compute for $C_{i,j}$, the dot product between row $i$ of $A$ and column $j$ of $B$. This is graphically depicted below: ![Fig1](/assets/Matrix_multiplication_svg.png)*Image taken from Wikipedia-Matrix Multiplication*


<u>Matrix vector multiplication</u>
* Linear Maps
* System of Linear Equations, i.e. Linear Program with multiple linear constraints, represented by $m$ equations and $n$ variables. The general form of a system of linear equations is: 

$$
\begin{align}
a_{11}x_1 + a_{12}x_2 + ... a_{1n}x_n &= b_1 \\\
a_{21}x_1 + a_{22}x_2 + ... a_{2n}x_n &= b_2 \\\
... &= ... \\\
a_{m1}x_1 + a_{m2}x_2 + ... a_{2m}x_n &= b_m \\\
\end{align}
$$

This is equivalent to a Matrix product, $Ax=b$ where $A \in \mathbb{R}^{m\times n}$ represents the coefficients on the unknown, and itself has known values,  $x \in \mathbb{R}^{n\times 1}$ representing unknown variables that we are trying to solve for. $b \in \mathbb{R}^{m \times 1}$ and the $i$-th row of $Ax=b$ represents the $i$-th constraint. The goal is to solve for $x$.


<u>Tensors</u>

Tensors are a X-D array of numbers. They are often used for datapoints which need to be represented in multiple dimensions, for example, a single image can be represented as a 3-dimensional Tensor', as it has height, width, and color channels.

**Solving $A^{-1}b$**

<u>Matrix Inversion and the Analytical Solution of $x=A^{-1}b$</u>

Matrix inversion allows us to solve equations of the form $Ax=b$ for many values of $A$. The inverse of $A$, denoted as $A^{-1}$ is defined as the matrix such that $A^{-1}A = I_n$. $I_n$ is an identity matrix that does not change any vector when we multiply the vector by it. An identity matrix that preserves $n$-dimensional vectors is $I_n$. Formally, $I_n \in \mathbb{R}^{n\times n}$ with diagonal entries equal to 1 and 0 everywhere else. We can now solve for $x$:

$$
\begin{align}
Ax &= b \\\
A^{-1}Ax &= A^{-1}b \\\
I_nx &= A^{-1}b \\\
x &= A^{-1}b
\end{align}
$$

In practice, $A^{-1}$ may not exist and the conditions for this is Linear Independence between the constraint vectors. When it does exist, several different algorithms can be used to find $A^{-1}$. 

**Matrices and Linear Transformations**
When dealing with data, we phrase things in terms of matrices and their associated linear transformations.

**Span, Linear Independence of a set of vectors, and the existence of a Closed-Form Analytical Solution**

The span of a set of vectors, $V = [v^1,  v^2, ..., v^i]$, where $i \in {1 .. \|V\|}$, is the space (or set of all points) that can be reached all linear combinations of these vectors. A linear combination of vectors means multiplying each vector $v^{i}$ by a scalar coefficient and summing over all vectors in $V$. 


\begin{equation}
Span(V) = \sum_i c_i v^i, for \ all \ c_i
\end{equation}

Linear dependence of the set means that one of the vectors can be expressed as some combination of the other vectors in the set. For example, the set of {[2, 3], [4, 6]} is linearly dependent. This is easy to see as [4, 6] is 2 * [2, 3]. Linearly independent vectors means that every vector must add some new 'directionality' which cannot be expressed by the other vectors. For example, [1,0, 0], [0, 1, 0], [0, 0, 1] are linearly independent in $\mathbb{R}$.

Linear dependence helps us understand how many dimensions our vector space actually has. A basis is a set of $n$ vectors, that are not linear combinations of each other (linearly independent). They thus span an $n$-dimensional space.


<u>Norms and Cost Functions </u>
Norms give us some way to quantify the 'size' of a vector or matrix, and these are often seen in cost/loss functions when we want to get some overall loss to minimize.

Two common norms (or loss functions) that we encounter are the L1, L2, squared L2, L$\infty$ (max norm).

Matrix norms that are analogous to vector norms are used when we consider optimising batches of inputs, instead of a single input.
For example, we want to find the parameters $W$ that minimise $y - Wx$ over all datapoints $X=\{x_1, \cdots x_n\}$. If we apply a squared L2 loss, then the matrix analog is the squared frobenius norm.

\begin{equation}
  min_W \sum_{i} || y_i - Wx_i ||_2^2 \\
  = min_W ||Y - WX ||_2^2
\end{equation}


<u>Dot Product(Inner Product) and Decision Rules</u>

The dot product of two vectors, $v_1$ and $v_2$, is denoted by 

\begin{equation}
v_1 . v_2 = \|v_1\| \|v_2\| cos(\theta)
\end{equation}

The dot product captures something about the direction of the two vectors with reference to each other via $\theta$, the angle between the two vectors. 
1. If the vectors are pointing in the same direction, then $\theta$ is 0, and $cos(\theta)=1$ and $v_1 . v_2 = \|v_1\|\|v_2\|$.
2. If the vectors are on the same side as straight line boundary, then $v_1 . v_2 > 0$.
3. If the vectors are on opposite sides as a lined boundary, then $v_1 . v_2 <0$.
2. If the vectors are orthogonal to each other, then $\theta$ is 90 deg, and $cos(\theta)=0$ and $v_1 . v_2 = 0$. 
3. If the vectors are pointing in exact opposite directions, then $\theta$ is 180, and $cos(\theta)=-1$, and $v_1 . v_2 = - \|v_1\| \|v_2\|$.

The scalar projection can be derived directly by manipulating the dot product equation. 

\begin{equation}
\frac{v_1.v_2}{\|v_1\|} = \|v_2\|cos(\theta)
\end{equation}

This is equivalent to the projection of $v_1$ onto $v_2$, because $\|v_2\|cos(\theta)$ is equivalent to the length of the 'adjacent side'. 

**Vector Projections, Basis vectors, Change of basis**

Any vector in the span of this set of basis vectors can be represented by a linear combination of the basis vectors. Note that we can use a different set of basis vectors, to represent the same point, but using a different linear combination of the new set of basis vectors. 

We use vector projections to change from one coordinate space to another coordinate space. If the new basis vectors are orthogonal to each other, we can use the projection (or dot product) to find out the vector coordinates in the new basis. If the basis vectors are not orthogonal, we need to do a transformation of axes using a matrix product which is computationally less efficient. To get the new vector coordinates in the new basis, for each of the new basis vectors, we do a scalar projection onto the new basis vectors, and then sum the projected vectors. 

Basis vectors which are orthogonal (90deg) to each other and are unit vectors, are called orthonormal basis vectors. 

Applications
- Test error for goodness of line fit
- Remapping data into feature rich space
- Change of Basis and Non-linearity

**Change of Basis and Non-linearity*
Non-linear bases and change of basis allows us to fit non-linear models. The trick is to transform the coordinate system into a higher dimensional space, where the regression or classification is linear. We can have a linear function of W but quadratic function of X. 

**Eigenvectors and Dimensionality Reduction - LDA and PCA**

<u>Why dimensionality reduction?</u>
- Dimensionality reduction is a common preprocessing technique in pattern classification and machine learning. 
- avoid the curse of dimensionality and overfitting
- reduce computational costs.

<u>How to achieve lower dimensionality without losing information?</u>

<u>What does this mean mathematically?</u>

The goal is to project a dataset from a high dimensional feature space to a lower-dimensional space. In the unsupervised setting when labels are unknown, this means finding the directions that maximise the variance in the data, a procedure known as PCA. In the supervised setting, when labels are known, this means finding the directions that maximises the separation between classes, a method known as LDA.

A good feature representation is found when the eigenvalues have a similar magnitude.

<u>PCA</u>

<u>LDA</u>
LDA has two parts, dimensionality reduction and classifier. In this section we will focus only on the dimensionality reduction aspect. To find directions that maximise the separation between classes, we wish to find the subspace which maximises the variance of class means after projection.

<u>Eigenvectors and Eigenvalues</u>

<u>Advantages of LDA as a classifier</u>
LDA has a closed-form solution that can be easily computed, and has no hyperparamters to tune.

* Determinant
* Subspaces and Multi-label classification
* Scalars, Vectors, Matrices, Tensors and Data Representation
* Matrix Inversion, Pseudo Inverse and the Analytical Solution
* Linear Independence, Span and existence of a Closed-Form Analytical Solution
* Dot Product and Data Point Similarity
* Projections, Eigenvectors and Dimensionality Reduction
* Eigendecomposition, matrix decomposition, eigenvalue, convexity
* Kernels, feature combinations, basis functions
* Matrix Factorization and SVD
* Hilbert Space and SVM
* Subspaces
* Basis functions
* Norms and Regularization
* Sparse matrices and Compressed Sensing
* Normal equations, Cholesky factorization and Linear Regression

<br><br>

### Model Preliminaries

\begin{equation}
\end{equation}

<br><br>

{% highlight python %}
{% endhighlight %}

#### References ####
[WikipediaReference](https://wikipedia.org)

