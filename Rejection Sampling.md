This text is based on Casella and Berger's (2024) section on Rejection sampling. I created the simulations using Desmos[^1].

Rejection sampling involves simulating the distribution of a random variable $X$ (called the _target distribution_) using a subset of the variates drawn from some _candidate distribution_. This text will focus on the visual intuition of rejection sampling, rather than the formal proof.

**Accept/Reject Algorithm**
>Let $Y \sim f(y)$ and $V \sim g(V)$, where $f$ (the target density) and $g$ (the candidate density) have a common support. Define $$M=\sup_{y}(f(y)/g(y))< \infty$$.
>To generate a random variable $Y \sim f$:
>1. Generate independent random variables $U \sim \mathcal{U}(0,1)$ and $V \sim g$.
>2. If $UMg(V)<f(V)$ then accept and set $Y=V$; otherwise reject and return to step 1.


**Example 1**: Suppose our target distribution is $X \sim \text{beta}(2.7,6.3)$. If the target distribution's parameters were integers, we could use inverse transform methods to generate variates (we will exploit this in a later example), the same is not true for non-integer parameters. Hence we use rejection sampling.

is Let $U,V \sim \mathcal{U}(0,1)$ and


## Simulation Links
1. [Generating a Beta(2.7,6.3) sample from a Uniform(0,1) sample](https://www.desmos.com/calculator/w1b5kn7ybg)
2. [Generating a Symmetric Truncated Normal(2,1) sample from a Uniform(0,1) sample](https://www.desmos.com/calculator/ku3vjyysj0)
3. [Generating a Beta(2.7,6.3) sample from a Beta(2,6) sample](https://www.desmos.com/calculator/gsfxnfqemm)

## Remarks
[^1]: Generally, one should only use a PRNG with a known generation algorithm. For example, the R function `runif()` uses a Mersenne Twister to generate variates, which suffices for most non-cryptographic purposes. However, Desmos is not transparent about the method used to compute random variates.

## References
Casella, G., and Berger, R.L. (2024). _Statistical inference_ [2nd ed.]. CRC Press.  \
