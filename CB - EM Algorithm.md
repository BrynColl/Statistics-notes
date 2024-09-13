# The EM Algorithm

## The Algorithm
>EM Algorithm
>We can find an MLE using the following method:
>1. Choose an initial guess $\theta=\theta_{0}$.
>2. Given the observed data $\mathbf{y}$, and under the assumption that your current guess $\theta=\theta_{t}$ is correct, compute how likely it is that the complete data is $\mathbf{X}=\mathbf{x}$. That is, compute $k_{t}(\mathbf{z})=f(\mathbf{z}|\mathbf{y},\theta_{t})$.
>3. Dispose of guess $\theta_{t}$ but keep the computed guess $k_{t}(\mathbf{z})$ of the complete data. 
>4. Use $k_{t}(\mathbf{z})$ to compute $Q(\theta,\theta_{t})=\mathbb{E}[\ell(\theta|\mathbf{y},\mathbf{z})]$.
>5. Make a new guess $\theta_{t+1}=\arg\max_{\theta}Q(\theta,\theta_{t})$.
>6. If $\lvert \theta_{t+1}-\theta_{t} \rvert>\tau$ for some threshold value $\tau$, then return to step 2 and repeat. Else, stop.

Steps (2)-(4) are collectively known as the Expectation, or E-step while step (5) is called the M-step. The EM estimate is guaranteed to never get worse, but is not guaranteed to be the global maximum if there are multiple local maxima. In practice, one usually starts with multiple initial guesses and chooses the estimate with the largest likelihood in the terminal step.

The intuition for the EM method is as follows. Suppose we have a complete data set with drawn from a distribution with some unknown parameter $\theta$. We can use the data set to estimate $\theta$ with $\hat{\theta}$ (using, for example, MLE). Conversely, suppose we have an incomplete data set with, say, one unknown data point, but we know the distribution. Then, we could use knowledge of $\theta$ and the distribution to guess the value of the missing data point. For example, if $\mathbf{x}=(0,1,2)$ and $\mathbf{X} \sim \mathcal{N}(\theta,1)$, then a reasonable guess would be $\hat{\theta}=\bar{x}=1$, since $\bar{X}$ is normally distributed with the same mean. Conversely, suppose $\mathbf{x}=(0,z,2)$ where $\mathbf{X} \sim \mathcal{N}(1,1)$. Then, since the distribution has the greatest density around the mean, $z=1$ would be a reasonable guess.

The EM method attempts to estimate $\theta$ in cases where both $\theta$ and some $\mathbf{z} \in \mathbf{x}$ are unknown. The motivations for using EM are twofold. Ideally we would like to directly estimate $\theta$ by maximizing the likelihood (or, more usually, the log-likelihood) of the complete data set $\mathbf{X}=(\mathbf{Y},\mathbf{Z})$, where $\mathbf{Y}$ gives the observed data. One issue is that because latent variable $\mathbf{Z}$ is unobservable, we don’t know its distribution, and therefore cannot directly calculate $f(\mathbf{x}|\theta)$. Instead, we must make do with the marginal pdf/pmf of the observed data, $f(\mathbf{y}|\theta)$. A second issue is that the marginal log-likelihood $\ell(\theta)=\log f(\mathbf{y}|\theta)$ is often analytically intractable, or requires us to perform a massive number of intermediate calculations.  The EM algorithm attempts to calculate $\hat{\theta}$ while resolving these two issues. First, EM uses Jensen’s inequality (as logarithms are concave) to find some analytically tractable lower bound (it is easier to differentiate a sum of logs than a log of sums) of the marginal log-likelihood. Maximizing will give some updated guesstimate that is no worse than the one we currently have (in that it produces a higher likelihood of generating the observed data), so if we continue this process iteratively, our estimates should *ideally* converge to an MLE, though in reality this doesn’t always happen.

## Notation
***Random variables***
$\mathbf{X}=(\mathbf{Y},\mathbf{Z})$ is the complete data set, where $\mathbf{Y}$ is the vector of observed variables and $\mathbf{Z}$ is the vector of unobservable, latent variables. The supports of $\mathbf{X}$ and $\mathbf{Z}$ are respectively $\mathcal{X}$ and $\mathcal{Z}$. The unknown parameter we want to estimate is $\theta$. The EM algorithm is not Bayesian as otherwise $\theta$ would be a latent random variable like $\mathbf{Z}$.

***Density/mass functions***
The (unknown) joint pdf/pdf of $\mathbf{X}$ is $\mathbf{x}\mapsto f_{\mathbf{X}}(\mathbf{y},\mathbf{z})=f(\mathbf{y},\mathbf{z}|\theta)$. We will abbreviate the joint function as $f(\mathbf{y},\mathbf{x})$. The associated likelihood and log-likelihood functions are respectively $\theta \mapsto f(\mathbf{y},\mathbf{z}|\theta)=L(\theta|\mathbf{y},\mathbf{z})$ and $\ell(\theta|\mathbf{y},\mathbf{z})=\log L(\theta|\mathbf{y},\mathbf{z})$.

The marginal pdf/pmf is $\mathbf{y} \mapsto f_{\mathbf{Y}|\theta}(\mathbf{y})=f(\mathbf{y}|\theta)$ and is abbreviated as $f(\mathbf{y})$. By marginalization, we have $f(\mathbf{y})=\int_{\mathcal{Z}} \, f(\mathbf{y},\mathbf{z}) d\mathbf{z}$ (interpret the integral as a generalized weighted sum throughout this section). The associated likelihood and log-likelihood functions are respectively $\theta \mapsto f(\mathbf{y}|\theta)=L(\theta|\mathbf{y})$ and $\ell(\theta|\mathbf{y})=\log L(\theta|\mathbf{y})$. We abbreviate the log-likelihood function as $\ell(\theta|\mathbf{y})=\ell(\theta)$.

The (unknown) conditional pdf/pmf of $\mathbf{Z}$ given $\mathbf{Y}=\mathbf{y}$ is $\mathbf{z} \mapsto k(\mathbf{z}|\mathbf{y},\theta)$, which we will sometimes abbreviate as $k(\mathbf{z})$ or simply $k$. Note that we are *given* the data $\mathbf{Y}=\mathbf{y}$ but treat $\theta$ as some fixed parameter with an implied value. The distribution of $\mathbf{Z}$ is unknown until we know the value of $\theta$ (conversely, the value of $\theta$ is unknown until we know the distribution of $\mathbf{Z}$). We relate the three pdfs/pmfs by:

```math
$$
k(\mathbf{z})=\frac{f(\mathbf{y},\mathbf{z})}{f(\mathbf{y})}
$$
```

Often, we are concerned with the distribution of $\mathbf{Z}$ given $\mathbf{Y}=\mathbf{y}$ when we set $\theta$ equal to some estimate $\theta_{t}$. We will call this random variable “$\mathbf{Z}$ given $\mathbf{y}$ under $\theta=\theta_{t}$” and denote it as $\mathbf{Z}|\mathbf{y};\theta_{t}$. The associated pdf/pmf is $k(\mathbf{z}|\mathbf{y},\theta_{t})$, which differs from $k(\mathbf{z}|\mathbf{y},\theta)$ in that it substitutes the unknown parameter $\theta$ with the known estimate $\theta_{t}$. We will use the abbreviations $k_{t}=k_{t}(\mathbf{z})=k(\mathbf{z}|\mathbf{y},\theta_{t})$. 

In summary: (breezes)
- $\mathbf{X} \sim f(\mathbf{y},\mathbf{z})$
- $\mathbf{Y} \sim f(\mathbf{y})$
- $\mathbf{Z}|\mathbf{y} \sim k(\mathbf{z})$
- $\mathbf{Z}|\mathbf{y};\theta_{t} \sim k_{t}(\mathbf{z})$

Lastly, the notation $\mathbb{E}_{k_{t}}[.]$ means we are taking the expectation of the argument with respect to the subscript. In symbols:

```math
$$
\mathbb{E}_{k_{t}}[h(\mathbf{z})]=\int_{\mathcal{Z}} k_{t}(\mathbf{z}|\mathbf{y},\theta_{t})h(\mathbf{z})\, d\mathbf{z}
$$
```

Naturally, if $h$ is $\mathbf{z}$-invariant, we would conclude that $\mathbb{E}_{k_{t}}[h(.)]=h(.)$.

It may also be useful to impress some information theory concepts into statistical service. We define the following functions:

```math
$$
\begin{align}
H(k_{t}) =H(\theta_{t}|\theta_{t})&\overset{\Delta}{=} -\mathbb{E}_{k_{t}}[\log k_{t}(\mathbf{z})] \\
H(k_{t},k)=H(\theta|\theta_{t}) &\overset{\Delta}{=} -\mathbb{E}_{k_{t}}[\log k(\mathbf{z})] \\
D_{KL}(k_{t},k) & \overset{\Delta}{=} H(k_{t},k)-H(k_{t}) \\
Q(\theta|\theta_{t}) &\overset{\Delta}{=} \mathbb{E}_{k_{t}}[\log f(\mathbf{y},\mathbf{z})]
\end{align}
$$
```

The first function is the **entropy** of $\mathbf{Z}|\mathbf{y};\theta_{t}$ (that is, $\mathbf{Z}|\mathbf{y}$ when we set $\theta=\theta_{t}$). It is the average amount of information we would obtain from observing the realizations of $\mathbf{Z}|\mathbf{y}$ under $\theta=\theta_{t}$. The second function is the **cross entropy** of the distribution of $\mathbf{Z}|\mathbf{y}$ relative to the distribution of $\mathbf{Z}|\mathbf{y};\theta_{t}$. I know jack shit about information theory, so I don’t know what it means outside of its name. We relate entropy and cross entropy by the **Kullback-Leibler (KL) divergence** of $k_{t}$ and $k$, which measures the similarity between the two probability distributions (in particular, it quantifies the expected amount of additional information we would get from using $k$ as a model instead of $k_{t}$). **Gibbs’ inequality** argues (via Jensen) that the KL divergence is non-negative for any pair of distributions $(P,Q)$, and equals zero iff $P=Q$. Finally, the fourth function is creatively called the **Q-function**. AFAIK it is native to EM and is unrelated to the tail distribution of $\mathcal{N}(0,1)$. It plays a role in defining a fifth function, called the **Evidence Lower Bound (ELBO)**

```math
$$
\begin{align}
\text{ELBO}(\theta,q) & =\mathbb{E}_{q}[\log f(\mathbf{y},\mathbf{z}|\theta)]+H(q(\mathbf{z})) \\
 & =\mathbb{E}_{q}[\log f(\mathbf{y}|\theta)]-D_{KL}(q(\mathbf{z}), k(\mathbf{z}))
\end{align}
$$
```

where $q=q(\mathbf{z})$ is an arbitrary pdf/pmf with support $\mathcal{Z}$. Since $f(\mathbf{y})$ is $\mathbf{z}$-invariant, we can rewrite the second line as:

```math
$$
\text{ELBO}(\theta ,q)=\log f(\mathbf{y})-D_{KL}(q,k)
$$
```

If $q(\mathbf{z})=k(\mathbf{z})$, then the expectation in the first line would become $Q(\theta|\theta_{t})$. ELBO’s name derives from the non-negativity of KL divergence, and the interpretation of $\log f(\mathbf{y})$ as the **evidence for $\mathbf{y}$**. 

The ML algorithm boils down to optimizing the two-variable ELBO function using **coordinate ascent**:
1. Fix an arbitrary guesstimate $\theta=\theta_{0}$.
2. Use $\theta_{0}$ to obtain $q_{0}=\arg\max_{q}\text{ELBO}(\theta_{0},q)$.
3. Release $\theta$ and fix $q=q_{0}$ to obtain $\theta_{1}=\arg\max_{\theta}\text{ELBO}(\theta,q_{0})$.
4. If $|\theta_{1}-\theta_{0}|$ is below some threshold, end the process and use $\theta_{1}$ as an estimate. Otherwise, iterate the algorithm.


## Why does it work?
Well to begin with, EM isn’t guaranteed to work. That is, while we can prove that the EM log-likelihoods are monotonically non-decreasing (that is, our estimates do not get worse), EM does not ensure that these EM likelihoods convergence to a global maximum.

>Gibbs’ Inequality
>If $F$ and $G$ are any two distribution functions with common support $R$, and respective pdfs/pmfs $f$ and $g$, then: $$D_{KL}(f,g)\geq 0$$ and $D_{KL}(f,g)= 0$ iff $f(x)=g(x)$.


PROOF:
The function $\log\left( \frac{1}{U} \right)=-\log U$ is  convex in $U$. If $U=g(X)/f(X)$ then, by Jensen's inequality:

```math
$$
\begin{align}
\mathbb{E}_{f}[-\log U]&\geq -\log \mathbb{E}_{f}[U] \\
-\mathbb{E}_{f}\left[ \log \frac{g(X)}{f(X)} \right] &\geq -\log \mathbb{E}_{f}\left[ \frac{g(X)}{f(X)} \right] \\
-\mathbb{E}_{f}[\log g(X)]+\mathbb{E}_{f}[\log f(X)]&\geq -\log\left( \int_{R} f(x)\frac{g(x)}{f(x)} \, dx \right)  \\
H(f,g)-H(f) &\geq -\log 1 \\
D_{KL}(f,g) &\geq 0
\end{align}
$$
```
$\blacksquare$


>Theorem
>The sequence of log-likelihood functions $(\ell(\theta_{t}))_{t}$ is monotonically non-decreasing for all steps $t$ of the EM algorithm.

PROOF
We wish to show that $\ell(\theta_{t+1})\geq \ell(\theta_{t})$, where $\theta_{t+1}=\arg\max_{\theta}\text{ELBO}(\theta,q_{t})$ and $q_{t}=\arg\max_{q}\text{ELBO}(\theta_{t},q)$. 

Since $\text{ELBO}(\theta,q)=\mathbb{E}_{q}[\log f(\mathbf{y})]-D_{KL}(q,k)$ where $\mathbb{E}_{q}[f(\mathbf{y})]=f(\mathbf{y})$ is $\mathbf{z}$-invariant, it follows that:
```math
$$
\arg\max_{q} \text{ELBO}(\theta_{t},q)=\arg\min_{q}D_{KL}(q,k_{t})
$$
```

Then, by Gibbs, $q_{t}=k_{t}$ must be the maximizer in step $t$. Then, $\theta_{t+1}$ maximizes:

```math
$$
\begin{align}
\text{ELBO}(\theta,q_{t}) & =\text{ELBO}(\theta,k_{t}) \\
 & = \mathbb{E}_{k_{t}}[\log f(\mathbf{y},\mathbf{z}|\theta)]+H(k_{t}) \\
 & = Q(\theta|\theta_{t})+H(k_{t})
\end{align}
$$
```

If $\theta_{t+1}$ maximizes $\text{ELBO}(\theta,k_{t})$, we have $\max\{ Q(\theta_{t+1}|\theta_{t}),Q(\theta_{t}|\theta_{t}) \}$, so $Q(\theta_{t+1}|\theta_{t})\geq Q(\theta_{t}|\theta_{t})$.

Since $\ell(\theta)$ is $\mathbf{z}$-invariant, we can write:

```math
$$
\begin{align}
\ell(\theta) & =\mathbb{E}_{k_{t}}[\ell(\theta)] \\
 & = \mathbb{E}_{k_{t}}[\log f(\mathbf{y}|\theta)] \\
 & = \mathbb{E}_{k_{t}}\left[ \log\left( \frac{f(\mathbf{y},\mathbf{z}|\theta)}{k(\mathbf{z}|\mathbf{y},\theta)} \right) \right] \\
 & = Q(\theta|\theta_{t})+H(k_{t},k) \\
\end{align}
$$
```
Consequently,

```math
$$
\ell(\theta_{t+1})-\ell(\theta_{t})= [Q(\theta_{t+1}|\theta_{t})-Q(\theta_{t}|\theta_{t})]+[H(k_{t},k_{t+1})-H(k_{t})]
$$
```

We already know that the difference in the Q-functions is non-negative. Furthermore, Gibbs’ inequality implies that $H(k_{t},k_{t+1})\geq H(k_{t})$. Therefore,

```math
$$
\ell(\theta_{t+1})\geq \ell(\theta_{t})
$$
```
$\blacksquare$

## Sources
Casella, G., and Berger, R.L. (2001). Statistical inference [2nd ed.]. Cengage.
https://en.m.wikipedia.org/wiki/Expectation–maximization_algorithm
https://teng-gao.github.io/blog/2022/ems/
https://www.statlect.com/fundamentals-of-statistics/EM-algorithm#:~:text=The%20Expectation%2DMaximization%20(EM),(also%20called%20latent%20variables)
[https://www.math.kth.se/matstat/gru/Statistical%20inference/Lecture8.pdf](https://www.math.kth.se/matstat/gru/Statistical%20inference/Lecture8.pdf) [https://myweb.uiowa.edu/pbreheny/7110/wiki/kullback-leibler.html](https://myweb.uiowa.edu/pbreheny/7110/wiki/kullback-leibler.html) [https://myweb.uiowa.edu/pbreheny/7110/wiki/gibbs-inequality.html](https://myweb.uiowa.edu/pbreheny/7110/wiki/gibbs-inequality.html)
https://www.youtube.com/watch?v=xy96ArOpntA
