---
layout:     post
title:      "Quiz answers for 16-385 Computer Vision"
subtitle:   "Full of stars"
date:       2023-03-30 12:00:00
author:     "Yubin"
header-img: "img/Headers/mio.jpg"
mathjax: true
catalog: true
tags:
    - Computer Vision
---

# Answers

This note is my answers of 16-385 Computer Vision take-home quiz. I'd appreciate it if you could point out the errors in my answers. If there are any mistakes, please send an email to yubinliu925@gmail.com . I will not type the original questions, because you can find them on the [course website](http://www.cs.cmu.edu/~16385/). I jumped the quiz 7 because I think it is quiet difficult for self-studied students.


## 1. Quiz 1

### Question 1

Actually, when $X$ and $Y$ are mutually independent random variables, the distribution function of $X+Y$ is the convolution of density functions $f_x$ and $f_y$:

$$
F_{X+Y}(a) = \iint_{X+Y\leq a}f_X(x)f_Y(y)dxdy=
\int_{-\infty}^{+\infty}\int^{a-y}_{-\infty}f_X(x)dxf_Y(y)dy \\
f_{X+Y}=\frac{d}{da}\int_{-\infty}^{+\infty}F_X(a-y)f_Y(y)dy = \int_{-\infty}^{+\infty}
f_X(a-y)f_Y(y)dy
$$

Without prove, we have that the sum of two mutually independent Gaussian variables  $N_i\sim(\mu_i, \sigma^2_i)$ is also a Gaussian variable which satisfy:

$$
\mu = \mu_1 + \mu_2 \quad \sigma^2 = \sigma_1^2+\sigma_2^2
$$

So the convolution of one Gaussian with another produces a third Gaussian with scale equal to their sum. For more details, you can check [the Wikipedia](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables).

### Question 2

We can use Lagrange polynomial to approximate the function, and the error of the approximation is $O(h^n)$, where $n$ is the number of points. 

Using five points $(x_{i}, f(x_{i}))$ where $i=\{[-2, -1, 0, 1, 2]\}$, we have:

$$
\begin{aligned}
f'(x) &\approx L'(x_0) + O(h^4) \\&= 
\frac{2h^3}{24h^4}f(x_{-2}) + \frac{4h^3}{-6h^4}f(x_{-1}) + \frac{0h^3}{4h^4}f(x_0) + 
\frac{-4h^3}{-6h^4}f(x_1) + \frac{-2h^3}{24h^4}f(x_2) + O(h^4) \\
&= \frac{1}{12h}(f(x_{-2}) - 8f(x_{-1}) + 8f(x_{1}) - f(x_{2})) + O(h^4)
\end{aligned}
$$

where $L(x)$ is the Lagrange polynomials function using these five points.

So the corresponding convolution kernel is:

$$
\frac{1}{12} \left [ \begin{matrix} 
-1 & 8 & 0 & -8 & 1
\end{matrix} \right] 
$$


## 2. Quiz 2

### Question 1

For the following covariance metric:

$$
E_w(u,v;x,y) = \sum_{s,t}w(s,t)[I(x-s+u, y-t+v)-I(x-s,y-t)]^2
$$

Because the $u$ and $v$ are small perturbations, we can rewrite the formula as:

$$
\begin{aligned}
E_w(u,v;x,y) &= \sum_{s,t}w(s,t)[I(x-s+u, y-t+v)-I(x-s,y-t)]^2 \\
&\approx \sum_{s,t} \left [ \begin{matrix} u & v \end{matrix} \right] 
\left [ \begin{matrix} I_x(x-s,y-t) \\
 I_y(x-s,y-t) \end{matrix} \right] 
\left [ \begin{matrix} I_x(x-s,y-t) & I_y(x-s,y-t) \end{matrix} \right] 
\left [ \begin{matrix} u \\ 
v \end{matrix} \right] \\
&= \left [ \begin{matrix} u & v \end{matrix} \right] \cdot
\mathcal{M}_w(x,y) \cdot \left [ \begin{matrix} u & v \end{matrix} \right]^T
\end{aligned} 
$$

where $\mathcal{M}_w(x,y)$ is the covariance matrix:

$$
\mathcal{M}_w(x,y) = \sum_{s,t}w(s,t)
\left [ \begin{matrix} 
I_x(x-t,y-t)I_x(x-t,y-t) & I_x(x-t,y-t)I_y(x-t,y-t) \\
I_y(x-t,y-t)I_x(x-t,y-t) & I_y(x-t,y-t)I_y(x-t,y-t)
\end{matrix} \right]
$$

Using the definition of convolution, the covariance matrix can be written equivalently as:

$$
\mathcal{M}_w(x,y) = w(x,y) * 
\left [ \begin{matrix} 
I_x(x,y)I_x(x,y) & I_x(x,y)I_y(x,y) \\
I_y(x,y)I_x(x,y) & I_y(x,y)I_y(x,y)
\end{matrix} \right]
$$

The code of Harris Corner Detection in Python is:

```python
def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)
    
    dxx_conv = convolve(dx * dx, window, mode='constant', cval=0)
    dxy_conv = convolve(dx * dy, window, mode='constant', cval=0)
    dyy_conv = convolve(dy * dy, window, mode='constant', cval=0)
    
    R = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            M = np.array([[dxx_conv[i, j], dxy_conv[i, j]], 
                          [dxy_conv[i, j], dyy_conv[i, j]]])
            response[i, j] = np.linalg.det(M) - k * (np.trace(M) ** 2)

    return response
```


### Question 2

The answer mostly come from [this blog](https://www.milania.de/blog/Introduction_to_the_Hessian_feature_detector_for_finding_blobs_in_an_image), which is a very meticulous article focused on Hessian matrix in corner detection. It also has more details on scale normalization which can adapt different scales of blobs.

For the Hessian matrix :

$$
\mathcal{H}(x,y) = 
\left [ \begin{matrix} 
I_{xx}(x,y) & I_{xx}(x,y) \\
I_{xy}(x,y) & I_{yy}(x,y)
\end{matrix} \right]
$$

the eigenvector $e_1$ points in the direction of the highest curvature with the magnitude $\lambda_1$. Similarly, $e_2$ corresponds to the direction of lowest curvature with the strength of $\lambda_2$. Here are three cases to consider for the second order derivative:

1. We have a maximum in both directions, i.e. $\lambda_1>0$ and $\lambda_2>0$ .
2. We have a minimum in both directions, i.e. $\lambda_1<0$ and $\lambda_2<0$ .
3. We have a minimum in one and a maximum in the other direction, e.g. $\lambda_1>0$ and $\lambda_2<0$ .

If both eigenvalues are large, then the corresponding position is a corner. So one way to detect these cases based on the eigenvalues is by using the Gaussian curvature:

$$
K = \lambda_1\cdot \lambda_2 = det(H)
$$

The main observation is that this product is only large when both eigenvalues are large. We can now detect blobs at each image position by calculating the Hessian matrix via image derivatives, their eigenvalues and then the Gaussian curvature $K$. Wherever $K$ is high we can label the corresponding pixel position as a blob. 

The code of Hessian Corner Detection in Python is:

```python
def Hessian_matrix(img, window_size, sigma):
    """
    Compute Hessian corner response map. Follow the math equation
    R=Det(H)

    Args:
        img: Gray scale image of shape (H, W)
        window_size: size of the window function
        sigma: the param of the gaussian mask

    Returns:
        response: Hessian response image of shape (H, W)
    """
	
  H, W = img.shape
  response = np.zeros((H, W))

  # generate the convolution kernel
	g, g_1d, g_2d = gaussian_mask(sigma, window_size)

	# smoothen the image
	Igx, Igy = gaussian_conv(img, g)

	# convoluting Image with 1st and 2nd derivative of I
	Ix, Iy = gaussian_1derv_conv(Igx, Igy, g_1d)
	Ixx, Iyy, Ixy = gaussian_2derv_conv(Ix, Iy, g_2d)

	for i in range(S[0]):
		for j in range(S[1]):
			Hessian = np.array([[Ixx[i][j], Ixy[i][j]], [Ixy[i][j], Iyy[i][j]]])
			response[i, j] = np.linalg.det(Hessian)
	return response
```

## 3. Quiz 3

### Question 1

For the heterogeneous least-square problem, we have:

$$
\begin{aligned}
\textbf{x} &= arg\min_\textbf{x} ||\textbf{Ax}-\textbf{b}||^2 \\
&= arg\min_\textbf{x} \textbf{x}^T\textbf{A}^T\textbf{Ax} - 2\textbf{b}^T\textbf{Ax}
+ \textbf{b}^T\textbf{b}
\end{aligned}
$$

$$
\frac{\partial ||\textbf{Ax}-\textbf{b}||^2}{\partial \textbf{x}} = 
2\textbf{A}^T\textbf{Ax} - 2\textbf{A}^T\textbf{b} = 0 
$$

So then we can derive the $\textbf{x}$:

$$
\textbf{x} = (\textbf{A}^T\textbf{A})^{-1}\textbf{A}^T\textbf{b}
$$

The visual interpretation of the residue can be got from [this link](https://en.wikipedia.org/wiki/Linear_least_squares#/media/File:Linear_least_squares_example2.svg). When we fit a line in the image, the parameters of slope or intercept may become zero, thus we should use Hough transform instead of this method.

For the homogeneous least-squares  problem, we have:

$$
\mathscr{l} = arg\min_{\mathscr{l}}||\textbf{A}\mathscr{l}||^2 \\
s.t.\ ||\textbf{B}\mathscr{l}||^2 = 1
$$

where we have:

$$
\textbf{B} = \left [ \begin{matrix} 
1 & 0 & 0 \\ 0 & 1 & 0 
\end{matrix} \right]
$$

which means $\sqrt{a^2 + b^2} = 0$. Therefore, when choosing this constraint, the residuals are orthogonal to the line. For more details of homogeneous least-squares  problem, you can refer [this link](https://foto.aalto.fi/seura/julkaisut/pjf/pjf_e/2005/Inkila_2005_PJF.pdf).

The problem can be solved using the generalized eigenvalue decomposition.

For finding a point that comes 'closest' to lying at the lines common intersection, we prefer tp choose the homogeneous least-squares method, since the residuals are orthogonal to the line, which are the definition of the distance between point and lines.


### Question 2

By the definition, we can find the least squares estimation as:

$$
\arg \min_\textbf{c} \sum_{i=1}^N |ax_i^2 + bx_iy_i + cy_i^2 + dx_i + ey_i + f|^2
$$

It only need 5 points to determine a unique solution for $\textbf{c}$. For the proof, you can refer [this link](https://en.wikipedia.org/wiki/Five_points_determine_a_conic).

By making the substitutions $x\rightarrow x_1/x_3$ and $y\rightarrow x_2/x_3$, the conic can be write as:

$$
ax_1^2 + bx_1x_2 + cx_2^2 + dx_1x_3 + ex_2x_3 + fx_3^2 = 0
$$

So it can be formulated as $\textbf{x}^T\textbf{C}\textbf{x}=0$, where:

$$
\textbf{C} = \frac{1}{2}\left [
  \begin{matrix}
    2a & b & d\\
    b & 2c & e\\
    d & e & 2f
  \end{matrix} \right ]
$$

By applying a projective transformation $\textbf{x}_i'=\textbf{Hx}_i$, the transformed conic can be written as:

$$
\textbf{C}'=\textbf{H}^T\textbf{CH}
$$

, which is still a symmetric matrix.

## 4. Quiz 4

### Question 1

First, we define the camera coordinates and pixel coordinates, a 3D point in camera coordinates $\textbf{p}$ is :

$$
\textbf{p} = [x\ y\ z]^T
$$

, and the projection of $\textbf{p}$ to pixel coordinates $\textbf{q}$ is that:

$$
\textbf{q} = [u\ v\ 1]^T = \frac{1}{z}\cdot \textbf{Kp}
$$

, where $\textbf{K}$ is the intrinsic camera matrix:

$$
\textbf{K} = \left [
  \begin{matrix}
    f_x & 0 & c_x\\
    0 & f_y & c_y\\
    0 & 0 & 1
  \end{matrix} \right ]
$$

For a point $p$ in camera coordinates $a$ and $b$, we have:

$$
\left [ \begin{matrix}
    \textbf{p}_b \\
    1
\end{matrix} \right ] = 
\left [ \begin{matrix}
    \textbf{R}_{ba}& \textbf{t}_{ba} \\
    \textbf{0}& 1
\end{matrix} \right ]
\left [ \begin{matrix}
    \textbf{p}_a \\
    1
\end{matrix} \right ] \\
\textbf{p}_b = \textbf{R}_{ba}\textbf{p}_a + \textbf{t}_{ba}
$$

, where $R_{ba}$ is the orientation of $a$ coordinate in $b$, and $\textbf{t}_{ba}$ is the position of $a$ coordinate in $b$. After transfer into pixel coordinates, we have:

$$
z_b\textbf{K}_b^{-1}\textbf{q}_b = z_a\textbf{R}_{ba}\textbf{K}_a^{-1}\textbf{q}_a + \textbf{t}_{ba} \\
$$

$$
\begin{equation}
\textbf{q}_b = \frac{z_a}{z_b}\cdot \textbf{K}_b\textbf{R}_{ba}\textbf{K}_a^{-1}\textbf{q}_a + \frac{1}{z_b}\cdot \textbf{K}_b\textbf{t}_{ba}
\end{equation}
$$

Suppose we have a plane $\pi$ in $a$ coordinate with normal vector $\textbf{n}_a$ and distance to the original point $d_a$, for point $p$, we have:

$$
\textbf{n}^T_a\cdot \textbf{p} + d_a = 0
$$

Therefore, we have:

$$
z_a = \frac{-d_a}{\textbf{n}^T_a\textbf{K}^{-1}_a\textbf{q}_a}
$$

So we can derive that:

$$
\begin{align*}
\textbf{q}_b &= \frac{z_a}{z_b}\cdot \textbf{K}_b\textbf{R}_{ba}\textbf{K}_a^{-1}\textbf{q}_a + \frac{1}{z_b}\cdot \textbf{K}_b\textbf{t}_{ba} \\
&= \frac{z_a}{z_b}\textbf{K}_b(\textbf{R}_{ba}\textbf{K}_a^{-1}\textbf{q}_a - \textbf{t}_{ba}
\frac{\textbf{n}^T_a\textbf{K}^{-1}_a\textbf{q}_a}{d_a}) \\
&=  \frac{z_a}{z_b}\textbf{K}_b (\textbf{R}_{ba} - \frac{\textbf{t}_{ba}\textbf{n}^T_a}{d_a})
\textbf{K}^{-1}_a\textbf{q}_a
\end{align*}
$$

Since we are in homogeneous coordinates, we can remove the coefficient, then we have:

$$
\textbf{q}_b = \textbf{K}_b (\textbf{R}_{ba} - \frac{\textbf{t}_{ba}\textbf{n}^T_a}{d_a})
\textbf{K}^{-1}_a\textbf{q}_a
$$

This means we have a homographic matrix $\textbf{H}$ satisfy the equation. Here the $\textbf{P}_1$ and $\textbf{P}_2$ are denoted as $\textbf{K}_a$ and $\textbf{K}_b$. 

If there is pure rotation for the camera, with the equation 1, we can find that:

$$
\textbf{q}_b = \textbf{K}_b\textbf{R}_{ba}\textbf{K}_a^{-1}\textbf{q}_a
$$

If the camera is rotating about its center $\textbf{C}$, we have:

$$
\textbf{R}(2 \theta) = \textbf{R}(\theta)\textbf{R}(\theta)
$$

Therefore, we have:

$$
\textbf{H}^2 = \textbf{K}\textbf{R}(\theta)\textbf{K}^{-1}\textbf{K}\textbf{R}(\theta)\textbf{K}^{-1} = \textbf{K}\textbf{R}(2\theta)\textbf{K}^{-1} 
$$

So $\textbf{H}^2$ is the homographic matrix corresponding to a rotation of $2\theta$.

If $p_0$ on $I_0$ be an image captured by a camera of point $p$, and $p_1$ on $I_1$ be an image of $I_0$ captured by another camera, we have:

$$
\textbf{p}_2 = \textbf{K}_2\textbf{p}_1 =  \textbf{K}_2\textbf{K}_1\textbf{p} 
$$

If $p_2=\textbf{0}^T$ is the camera center of $I'$, because the null space of an $n \times n$ invertible matrix $\textbf{A}$ is empty, we have $\textbf{p} = \textbf{0}^T$. This means the apparent camera center of $I'$ is the same as that of $I_0$.

I am not sure about this question, if you want to know more, you can check the first question in [this link](https://blog.immenselyhappy.com/post/mvg-sol-6/#1).

### Question 2

When the camera location is known, we only need to estimate 8 parameters, which means we need 4 pairs. And when the camera location and complete orientation are known, we only need to estimate 5 intrinsic parameters, so we only need 3 pairs.

<!-- I didn't understand the second problem in this question. -->

## 5. Quiz 5

### Question 1
Suppose the two camera centers are denoted by $A$ and $B$, since they differ only by a translation of their origins along a direction that is parallel to either the $x$ or $y$ axis of the coordinate systems, we may find that line $AB$ is parallel to the camera image plane. Therefore, the epipolar line on both image plane and plane $ABP$ where $P$ is the object point is parallel with the line $AB$. Therefore the epipolar lines of a rectified pair are parallel to the axis of translation.

Since we have $
\textbf{E} = \textbf{R}[\textbf{t}_{\times}]$ and $
\textbf{R} = \textbf{I}
$, without loss of general, we assume the translation is parallel with the $x$ axis, then $\textbf{t} = [x_0\ 0\ 0]^T$. Therefore, we have:

$$
\textbf{E} = \left [
  \begin{matrix}
    0 & 0 & 0\\
    0 & 0 & -x_0\\
    0 & x_0 & 0
  \end{matrix} \right ]
$$

### Question 2

Since the coordinates of both $\textbf{x}$ and $\textbf{x}'$ in image coordinate is $[0\ 0\ 1]^T$, we have:

$$
\textbf{x}'^T\textbf{F}\textbf{x} = \textbf{0} \\
$$

Therefore, $\textbf{F}_{33}=0$

### Question 3

Using $\textbf{x}'^T\textbf{F}\textbf{x} = 0$, we have:

$$
\left [
  \begin{matrix}
    \textbf{x}_1^T\textbf{F}_{13}\\
    \textbf{x}_2^T\textbf{F}_{23}
  \end{matrix} \right ] \textbf{x}_3 = \textbf{0}
$$

If the two rows of the matrix are not linear independent, then we can't solve the inverse matrix, which means the point $\textbf{x}_3$ cannot be uniquely determined by this expression.

## 6.Quiz 6

### Question 1

We have:

$$
\omega_1 = \frac{dA}{D^2} \qquad \omega_2 = \frac{dA\cos^3 \alpha}{D^2}
$$

Using the solid angle above, we can find the irradiance $E$ onn the plain at points $X_1$ and $X_2$ as:

$$
E(X_1) = L\frac{dA}{D^2} \qquad E(X_2) = L\frac{dA\cos^4 \alpha}{D^2} \\
\frac{E(X_1)}{E(X_2)} = \frac{1}{\cos^4 \alpha}
$$

### Question 2

For Lambertian surface, $\forall \hat{\textbf{v}}_i$, $f_r(\hat{\textbf{v}}_i, \hat{\textbf{v}}_o)$ are the same. $obviously we have $\rho\geq 0$, we want to derive that $\rho \leq 1$,
which means we just need to prove:

$$
f_r(\hat{\textbf{v}}_i, \hat{\textbf{v}}_o) \leq \frac{1}{\pi}
$$

Using the conservation of energy, we have:

$$
\int_{\Omega_{out}}f_r(\hat{\textbf{v}}_i, \hat{\textbf{v}}_o)\cos\theta_{out}d\omega \leq 1 \\
\int_{\Omega_{out}}\cos\theta_{out}d\omega = \pi
$$

Then we can prove the formula above.

For a specular surface, as described in the slides, the BRDF is a delta function:

$$
f(\hat{\textbf{v}}_i, \hat{\textbf{v}}_o) = \delta(\hat{\textbf{v}}_s, \hat{\textbf{v}}_o)
$$

### Question 3

Suppose $i\in \{1,2,3\}$, We have :

$$
I_i = \frac{\rho}{\pi} \hat{\textbf{n}}_i^T\hat{\textbf{s}}
$$

Then we can solve $\rho \hat{\textbf{s}}$ using:

$$
\rho \hat{\textbf{s}} = \pi
\left [ \begin{matrix}
    \hat{\textbf{n}}_1^T\\
    \hat{\textbf{n}}_2^T\\
    \hat{\textbf{n}}_3^T
  \end{matrix} \right ]^{-1}
\left [ \begin{matrix}
    I_1\\
    I_2\\
    I_3
  \end{matrix} \right ]
$$

Since $\hat{\textbf{s}}$ is an unit vector, we can do normalization to find $\hat{\textbf{s}}$ and $\rho$.

### Question 4

Obviously, we have:

$$
E_1 = \frac{1}{\sqrt{\hat{\textbf{n}}^T\hat{\textbf{s}}_1}\sqrt{\hat{\textbf{n}}^T\hat{\textbf{v}}}} \qquad

E_2 = \frac{1}{\sqrt{\hat{\textbf{n}}^T\hat{\textbf{s}}_2}\sqrt{\hat{\textbf{n}}^T\hat{\textbf{v}}}}
$$


Since $\hat{\textbf{n}}$ is an unit vector, we can solve the equations above and get $\hat{\textbf{n}}$.

## 7. Quiz 7

We will not finish this quiz. Please refer some courses or materials of physics-based methods in vision. The questions in this quiz is quiet difficult for self-studied students.

## 8. Quiz 8

### Question 1

Obviously, the $\textit{p-norms}$:

$$
D_p(\textbf{x}, \textbf{y}) \equiv ||\textbf{x} - \textbf{y}||_p = 
(\sum_{k=1}^{d}|x_k-y_k|^p)^{1/p}
$$

for all values of $p\geq 1$ satisfy the non-negativity, reflexivity and symmetry. We just need to prove the triangle inequality:

$$
D(\textbf{x}, \textbf{y}) + D(\textbf{y}, \textbf{z}) \geq D(\textbf{x}, \textbf{z})
$$

Therefore, we just need to prove the Minkowski inequality:

$$
(\sum_{k=1}^{d}|a_k+b_k|^p)^{1/p} \leq (\sum_{k=1}^{d}|a_k|^p)^{1/p} + 
(\sum_{k=1}^{d}|b_k|^p)^{1/p}
$$

We can prove this theory using Holder's inequality. For more details, you can refer [this link](https://www.planetmath.org/proofofminkowskiinequality).

### Question 2

For clarity, let $v_i$ where $i\in{1,\cdots,k}$ denote the Voronoi sites and $x_a$ and $x_b$ be arbitrary points in a particular Voronoi cell associated with site $v_i$. When $x$ and $v_i \in \mathbb{R}^N$, We need to prove that:

$$
||tx_a + (1-t)x_b - v_1||_2 \leq ||tx_a + (1-t)x_b - v_i||_2 \\
s.t.\quad ||x_a - v_1||_2 \leq ||x_a - v_i||_2 \quad 
||x_b - v_1||_2 \leq ||x_b - v_i||_2
$$

where $i\in{2,\cdots,k}$ and $0\leq t\leq 1$.

From the constrains, we can get that:

$$
||x_a - v_1||_2 \leq ||x_a - v_i||_2 \\
||x_a||^2 + ||v_1||^2 - 2x_av_1 \leq ||x_a||^2 + ||v_i||^2 - 2x_av_i \\
||v_1||^2 - 2x_av_1 \leq ||v_i||^2 - 2x_av_i
$$

Similarly, we have:

$$
||v_1||^2 - 2x_bv_1 \leq ||v_i||^2 - 2x_bv_i
$$

We notice that:

$$
||tx_a + (1-t)x_b - v_1||_2 \leq ||tx_a + (1-t)x_b - v_i||_2 \\
\Longleftrightarrow v_1^2 - 2(tx_a + (1-t)x_b)v_1 \leq v_i^2 - 2(tx_a + (1-t)x_b)v_i \\
\Longleftrightarrow t(||v_1||^2 - 2x_av_1) + (1-t)(||v_1||^2 - 2x_bv_1) \leq
t(||v_i||^2 - 2x_av_i) + (1-t)(||v_i||^2 - 2x_bv_i)
$$

thus we prove the theory.

## 9. Quiz 9

### Question 1

If we have a linear activation function $f(x) = W'x + b'$, then a network with one hidden layer can be write as:

$$
\begin{aligned}
g(x) &= W_2(W'(W_1x + b_1) + b') + b_2 \\
&= W_2W'W_1x + W_2(W'b_1+b') + b_2
\end{aligned}
$$

So we can use a non hidden layer network with $W = W_2W'W_1$ and $b = W_2(W'b_1+b') + b_2$ to replace the original network.

For the sigmoid function $\sigma(x)$, we have:

$$
\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = (1-\sigma(x))\sigma(x)
$$

When the absolute value of $x$ increases, the gradient of the sigmoid activation function changing slowly. When the sigmoid is used as the activation function for many layers, it will suffer Vanishing Gradient Problem. For more details, you can refer [this link](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484).

For the $tanh$ function, we have:

$$
tanh(x) = \frac{1 - e^{-2x}}{1 + e^{2x}} = 
\frac{2\sigma(x) - 1}{2\sigma^2(x) - 2\sigma(x) + 1} \\
tanh'(x) = \frac{4e^{-2x}}{(1+e^{-2x})^2} = 1-tanh^2(x)
$$

The output range of sigmoid function is $(0, 1)$, and the range of the hyperbolic  tangent function is $(-1, 1)$. About which function is better in different cases, you can refer [this link](https://stats.stackexchange.com/questions/101560/tanh-activation-function-vs-sigmoid-activation-function).

### Question 2

The second-order expansion of $L$ around a point ${x}^t \in \mathbb{R}^d$ is:

$$
\begin{aligned}
L(x) &= L(x^t) + \nabla L(x^t)^T(x-x^t) + \frac{1}{2}(x-x^t)^T\textbf{H}(x^t)(x-x^t) \\
\end{aligned}
$$

Combine the gradient descent iteration equation and the Taylor expansion above, we have:

$$
\begin{aligned}
L(x^{t+1}) &= L(x^t) + \nabla L(x^t)^T(x^{t+1}-x^t) + 
    \frac{1}{2}(x^{t+1}-x^t)^T\textbf{H}(x^t)(x^{t+1}-x^t) \\
&= L(x^t) + \nabla L(x^t)^T(-\eta(t)\nabla L(x^t)) + 
    \frac{1}{2}(-\eta(t)\nabla L(x^t))^T\textbf{H}(x^t)(-\eta(t)\nabla L(x^t)) \\
&= \frac{1}{2}\eta(t)^2(\nabla L(x^t)^T\textbf{H}(x^t)\nabla L(x^t)) - 
     \eta(t)\nabla L(x^t)^T\nabla L(x^t) + L(x^t)
\end{aligned}
$$

So we can get that:

$$
\begin{aligned}
\eta(t) &= arg\min \frac{1}{2}\eta(t)^2(\nabla L(x^t)^T\textbf{H}(x^t)\nabla L(x^t)) - 
     \eta(t)\nabla L(x^t)^T\nabla L(x^t) \\
&= \frac{||\nabla L(x^t)||^2}{\nabla L(x^t)^T\textbf{H}(x^t)\nabla L(x^t)}
\end{aligned}
$$

If we calculate the derivative of the Taylor expansion directly, we can get:

$$
\nabla L(x) = \nabla L(x^t) + \textbf{H}(x^t)(x-x^t)
$$

Let $\nabla L(x^{t+1}) = 0$, we have：

$$
\nabla L(x^{t+1}) = \nabla L(x^t) + \textbf{H}(x^t)(x^{t+1}-x^t) = 0 \\
x^{t+1} = x^t - \textbf{H}^{-1}(x^t)\nabla L(x^t) 
$$

This alternative gradient-descent procedure is often called Newton’s algorithm. It need to calculate the inverse of the Hessian matrix. 

For choosing an optimization algorithm in different situations, you can refer [this link](https://machinelearningmastery.com/tour-of-optimization-algorithms/).

## 10. Quiz 10

### Question 1

Convolution is not naturally equivalent to some other transformations, such as changes in the scale or rotation of an image. Other mechanisms are necessary for handling these kinds of transformations. Actually it is the max pooling layer that introduces such invariants.

The convolution layer is translation equivalent instead of invariant.

### Question 2

After the ReLU layer, all the score will in the range $[0, +\infty]$, so the next $tanh$ layer will only has the input greater than zero. Therefore, the output of the network will always be greater than zero, which means the predicted label is $+1$. So the classification accuracy of the network is $50\%$.

### Question 3

We assume the flow is locally smooth, so the surrounding patch (say $5\times5$) has 'constant flow', which means:

$$
\left [ \begin{matrix} \textbf{I}_x(x, y) & \textbf{I}_y(x, y) \end{matrix} \right] 
\left [ \begin{matrix} u\\v\end{matrix} \right] 
= - \left.  \begin{matrix} \textbf{I}_t(x, y)\end{matrix} \right.
$$

for all the pixels $x,y$ in the patch $W$ of size $N\times N$.

So we have:

$$
\textbf{A} = 
\left [ \begin{matrix} \textbf{I}_x(x, y) & \textbf{I}_y(x, y) \end{matrix} \right] \qquad \textbf{b} = - \textbf{I}_t(x, y)
$$

With multiplication of $\textbf{A}$, we can solve this heterogeneous linear system:

$$
\textbf{A}^T\textbf{A}\textbf{V}_W = \textbf{A}^T\textbf{b} \\
\textbf{V}_W = (\textbf{A}^T\textbf{A})^{-1}\textbf{A}^T\textbf{b}
$$

where we have:

$$
\textbf{A}^T\textbf{A} = \mathcal{M}_W = 
\left [ \begin{matrix} 
\sum_{(x,y)\in W}I_x(x, y)I_x(x, y) & \sum_{(x,y)\in W}I_x(x, y)I_y(x, y) \\
\sum_{(x,y)\in W}I_y(x, y)I_x(x, y) & \sum_{(x,y)\in W}I_y(x, y)I_y(x, y) \\
\end{matrix} \right] 
$$

So solving this heterogeneous linear system in the least-squares sense requires inverting the covariance matrix.

Actually the Harris Corner Detector is $\textbf{A}^T\textbf{A}$. When the Lucas-Kanade optical flow works well, $\textbf{A}^T\textbf{A}$ should be invertible, and $\lambda_1$ and $\lambda_2$ should not be too small, which means the patch $W$ is a corner. Corners are regions with two different directions of gradient, so they are good places to compute flow.

## 11. Quiz 11

### Question 1

In the Lucas-Kanade (or forward-additive) image alignment algorithm, the loss function is

$$
\min_\textbf{p}\sum_\textbf{x}[I(\textbf{W}(\textbf{x};\textbf{p}))-T(\textbf{x})]^2
$$

The first-order Taylor expansion of the composite function $I(\textbf{W}(\textbf{x};\textbf{p}))$ with respect to $\textbf{p}$ around the value $\textbf{p}^t$ is:

$$
I(\textbf{W}(\textbf{x};\textbf{p})) \approx I(\textbf{W}(\textbf{x};\textbf{p}^t)) + \frac{\partial I(\textbf{W}(\textbf{x};\textbf{p}^t))}{\partial \textbf{p}^t} 
  (\textbf{p} - \textbf{p}^t)
$$

So the approximation for $I(\textbf{W}(\textbf{x};\textbf{p}^t))$ is:

$$
\begin{aligned}
I(\textbf{W}(\textbf{x};\textbf{p}^t + \Delta\textbf{p}^t)) &\approx 
I(\textbf{W}(\textbf{x};\textbf{p}^t)) + 
\frac{\partial I(\textbf{W}(\textbf{x};\textbf{p}^t))}{\partial \textbf{p}^t} 
\Delta \textbf{p}^t \\ &= 
I(\textbf{W}(\textbf{x};\textbf{p}^t)) + 
\frac{\partial I(\textbf{W}(\textbf{x};\textbf{p}^t))}
{\partial \textbf{W}(\textbf{x};\textbf{p}^t)} 
\frac{\partial \textbf{W}(\textbf{x};\textbf{p}^t)}{\partial \textbf{p}^t}
\Delta \textbf{p}^t \\ &= 
I(\textbf{W}(\textbf{x};\textbf{p}^t)) + 
\nabla \textbf{I}
\frac{\partial \textbf{W}(\textbf{x};\textbf{p}^t)}{\partial \textbf{p}^t}
\Delta \textbf{p}^t
\end{aligned}
$$

Therefore, the optimization problem can be rewritten as:

$$
\begin{aligned}
  \min_\textbf{p}\sum_\textbf{x}[I(\textbf{W}(\textbf{x};\textbf{p}))-T(\textbf{x})]^2 &=
  \min_{\Delta\textbf{p}^t}\sum_\textbf{x}[
  I(\textbf{W}(\textbf{x};\textbf{p}^t)) + \nabla \textbf{I}
  \frac{\partial \textbf{W}(\textbf{x};\textbf{p}^t)}{\partial \textbf{p}^t}
  \Delta \textbf{p}^t - T(\textbf{x})]^2 \\ &=
  \min_{\Delta\textbf{p}^t}\sum_\textbf{x}[ \nabla \textbf{I}
  \frac{\partial \textbf{W}(\textbf{x};\textbf{p}^t)}{\partial \textbf{p}^t}
  \Delta \textbf{p}^t - (T(\textbf{x}) - I(\textbf{W}(\textbf{x};\textbf{p}^t)))]^2 \\
  &=  \min_{\Delta\textbf{p}^t}
  \sum_\textbf{x}||\textbf{A} \Delta \textbf{p}^t - \textbf{b}||^2
\end{aligned}
$$

where we have:

$$
\textbf{A} = \nabla \textbf{I}
  \frac{\partial \textbf{W}(\textbf{x};\textbf{p}^t)}{\partial \textbf{p}^t}\\
\textbf{b} = T(\textbf{x}) - I(\textbf{W}(\textbf{x};\textbf{p}^t))
$$

Here we need to derive the solution of the least squares problem first, then we can solve the optimization problem. The least squares problem is solved by:

$$
\begin{aligned}
\hat{\textbf{x}} &= arg\min_\textbf{x}\sum_i||\textbf{A}_i\textbf{x}-\textbf{b}_i||^2 \\
&= arg\min_\textbf{x} \sum_i\textbf{x}^T\textbf{A}_i^T\textbf{A}_i\textbf{x} - 
2\textbf{b}^T_i\textbf{A}_i\textbf{x} + \textbf{b}_i^T\textbf{b}_i \\
&=  arg\min_\textbf{x} \textbf{x}^T (\sum_i\textbf{A}_i^T\textbf{A}_i)\textbf{x} -
2(\sum_i\textbf{b}^T_i\textbf{A}_i)\textbf{x} + \sum_i\textbf{b}_i^T\textbf{b}_i
\end{aligned}
$$

Then we have:

$$
\frac{\partial \sum_i||\textbf{A}_i\textbf{x}-\textbf{b}_i||^2}{\partial \textbf{x}} = 
2\textbf{x}^T (\sum_i\textbf{A}_i^T\textbf{A}_i) - 2(\sum_i\textbf{b}^T_i\textbf{A}_i) = 0
\\ \textbf{x} = (\sum_i\textbf{A}_i^T\textbf{A}_i)^{-1}(\sum_i\textbf{A}_i^T\textbf{b})
$$

Using the equation above, we can derive the solution of the optimization problem:

$$
\Delta \textbf{p}^t = (\sum_\textbf{x}
  [\nabla \textbf{I} \frac{\partial \textbf{W}}{\partial \textbf{p}^t}]^T
  [\nabla \textbf{I} \frac{\partial \textbf{W}}{\partial \textbf{p}^t}])
  (\sum_\textbf{x}[\nabla \textbf{I} \frac{\partial \textbf{W}}{\partial \textbf{p}^t}]^T
  [T(\textbf{x}) - I(\textbf{W}(\textbf{x};\textbf{p}^t))])
$$

The algorithm procedure is:
1. Warp image $I(\textbf{W}(\textbf{x};\textbf{p}))$
2. Compute error image $[T(\textbf{x}) - I(\textbf{W}(\textbf{x};\textbf{p}^t))]$
3. Compute gradient $\nabla I(\textbf{x})$
4. Evaluate Jacobian $\frac{\partial \textbf{W}}{\partial \textbf{p}}$
5. Compute Hessian $\textbf{H} = \sum_\textbf{x}
  [\nabla \textbf{I} \frac{\partial \textbf{W}}{\partial \textbf{p}^t}]^T
  [\nabla \textbf{I} \frac{\partial \textbf{W}}{\partial \textbf{p}^t}]$
6. Compute $\Delta \textbf{p}$
7. Update parameters $\textbf{p} \leftarrow \textbf{p} + \Delta \textbf{p}$

### Question 2

The first-order Taylor expansion of the composite function $I(\textbf{W}^t(\textbf{W}(\textbf{x}, \Delta \textbf{p})))$ with respect to $\Delta \textbf{p}$ around the value $0$ is:

$$
I(\textbf{W}^t(\textbf{W}(\textbf{x}, \textbf{p}))) \approx
I(\textbf{W}^t(\textbf{W}(\textbf{x}, 0))) + 
\frac{\partial I(\textbf{W}^t(\textbf{W}(\textbf{x}, \textbf{p})))}{\partial \textbf{p}}
\Delta \textbf{p}
$$

Similarly as the question 1, the optimization problem can be rewritten as the least-squares form:

$$
\begin{aligned}
  &\min_{\Delta \textbf{p}^t}\sum_\textbf{x}[I(\textbf{W}^t(\textbf{W}(\textbf{x}, \Delta \textbf{p}))) - T(\textbf{x})]^2 \\ =& \min_{\Delta \textbf{p}^t}
  \sum_\textbf{x}[ I(\textbf{W}(\textbf{x};\textbf{p})) + \nabla \textbf{I}(\textbf{W})
  \frac{\partial \textbf{W}(\textbf{x};0)}{\partial \textbf{p}}
  \Delta \textbf{p}^t - T(\textbf{x}) ]^2 \\ =& \min_{\Delta \textbf{p}^t}
  \sum_\textbf{x} ||\textbf{A}\Delta\textbf{p}^t-\textbf{b}||^2
\end{aligned}
$$

where we have:

$$
\textbf{A} = \nabla \textbf{I}(\textbf{W})
  \frac{\partial \textbf{W}(\textbf{x};0)}{\partial \textbf{p}} \Delta \textbf{p}^t\\
\textbf{b} = T(\textbf{x}) - I(\textbf{W}(\textbf{x};\textbf{p}))
$$

The solution and algorithm procedure is also similarly as question 1. However, the Jacobian matrix $\frac{\partial \textbf{W}(\textbf{x},\textbf{0})}{\partial {\textbf{p}}}$ can be precomputed, which can speed up the algorithm.

### Question 3

The first-order Taylor expansion of the composite function $T(\textbf{W}(\textbf{x}, \Delta \textbf{p}))$ with respect to $\Delta \textbf{p}$ around the value $0$ is:
$$
T(\textbf{W}(\textbf{x}, \Delta \textbf{p})) \approx
T(\textbf{W}(\textbf{x}, \textbf{0})) + 
\frac{\partial T(\textbf{W}(\textbf{x}, \textbf{p})))}{\partial \textbf{p}}
\Delta \textbf{p}
$$

Similarly as the question 1, the optimization problem can be rewritten as the least-squares form:

$$
\begin{aligned}
  &\min_{\Delta \textbf{p}^t}\sum_\textbf{x}[I(\textbf{W}^t(\textbf{x})) - 
  T(\textbf{W}(\textbf{x},\Delta \textbf{p}^t))]^2 \\ =& 
  \min_{\Delta \textbf{p}^t}\sum_\textbf{x}[ 
   T(\textbf{W}(\textbf{x}, \textbf{0})) + \nabla T \frac{\partial \textbf{W}}
   {\partial \textbf{p}} \Delta \textbf{p} - I(\textbf{W}(\textbf{x},\textbf{p}))
   ]^2 \\ =& \min_{\Delta \textbf{p}^t} \sum_\textbf{x}
  ||\textbf{A}\Delta\textbf{p}^t-\textbf{b}||^2
\end{aligned}
$$

where we have:

$$
\textbf{A} = \nabla T \frac{\partial \textbf{W}}{\partial \textbf{p}}\\
\textbf{b} = I(\textbf{W}(\textbf{x};\textbf{p})) - T(\textbf{W}(\textbf{x};0))
$$

The solution and algorithm procedure is also similarly as question 1.  However, the Jacobian matrix, the gradient of template and the Hessian matrix can be precomputed, which makes this algorithm become the most efficient one.




