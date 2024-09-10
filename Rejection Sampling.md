# Visualizing Rejection Sampling

This text is based on Casella and Berger's (2001) section on Rejection sampling. I created the simulations using Desmos[^1] and R.

Rejection sampling involves simulating the distribution of a random variable $X$ (called the _target distribution_) using a subset of the variates drawn from some _candidate distribution_. This text will focus on the visual intuition of rejection sampling, rather than the formal proof.


### Rejection Sampling
Let $Y \sim f(y)$ and $V \sim g(V)$, where $f$ (the target density) and $g$ (the candidate density) have a common support. Define $$M=\sup_{x}(f(x)/g(x))< \infty$$. To generate a random variable $X \sim f$:
1. Generate independent random variables $U \sim \mathcal{U}(0,1)$ and $V \sim g$.
2. If $U<f(V)/Mg(V)$, then accept and set $X=V$; otherwise reject and return to step 1.

#### Example 1
Suppose our target distribution is $X \sim \text{beta}(2.7,6.3)$. If the target distribution's parameters were integers, we could use inverse transform methods to generate variates (we will exploit this in a later example), the same is not true for non-integer parameters. Hence we use rejection sampling. First, let $U \sim \mathcal{U}(0,1)$. In Desmos, we create a list uniform variates using `U = uniformdist().random(n)`, where the size of the list, `n`, is a slider-defined integer parameter. We define the candidate variates, `V`, in like manner. Next, we compute $M$. 
```math
$$
\begin{align}
M&=\sup_{x}\left( \frac{f(x)}{g(x)} \right) \\
 & = \sup_{x}\left( \frac{x^{1.7}(1-x)^{5.3}}{B(2.7,6.3)} \right) \\
 & = f\left( \frac{2.7-1}{2.7+6.3-2} \right) \\
 & \approx 2.67
\end{align}
$$
```
The envelope will be $Mg(x)$, which is just $M$ in our case. We simulate the "accept/reject" process by partitioning the variate plot $(V,Mg(V)U)$ into two sub-plots, our ultimate goal being to simulate the target distribution using the candidate variates. The first sub-plot consists of all the samples that fall underneath the target density. Such samples satisfy $U<f(V)/Mg(V)$ and are denoted by blue dots in the subsequent graphs. The second sub-plot consists of all the samples that fall between the envelope and the density curve and fail to satisfy $U<f(V)/Mg(V)$. We partition our data in desmos by partitioning `(V,Mg(V)U)` into the "accepted" sub-plot `(V,Mg(V)U{U<f(V)/Mg(V)})` and the "rejected" sub-plot `(V,Mg(V)U{U>=f(V)/Mg(V)})`.
<p>&nbsp;</p> 

![rejection_sampling_beta_by_unif](https://github.com/user-attachments/assets/16245b77-74a0-4fb2-b709-f4cf7de93219)
<div align="center">
 <i><b>Figure 1</b>: Uniform candidate density in Desmos, n = 200.</i>
</div>
<p>&nbsp;</p> 

We can also use the R code:
```
set.seed(42)
u <- runif(100,0,1)
v <- runif(100,0,1)
help("set.seed")

#Target pdf; note x is between 0 and 1
target <- function(x,a,b) {
  ((x^(a-1))*((1-x)^(b-1)))/beta(a,b)
}

#Candidate pdf
candidate <- function(x){
  if (0 <= x && x <= 1){
    1
  }
  else{
    0
  }
}
candidate.v <- Vectorize(candidate)

#M = sup(f/g), where g is the candidate density
x_M <- function(a,b){
  a <- 2.7
  b <- 6.3
  (a-1)/(a+b-2)
}

M <- function(x){
  target(x_M(x),2.7,6.3)
}

#Envelope, m*g; since g = 1 on (0,1), the envelope is just m on the interval
envelope <- Vectorize(M)

#plots
x.n <- seq(0,1,by=0.001)
plot(x.n,candidate.v(x.n),
     col="red",
     lty = "dashed",
     type='l',
     ylim = c(0,4),
     xlab = "v",
     ylab = "Mg(v)u"
     )
lines(x.n,envelope(x.n),
      col="red", 
      ylim = c(0,4))
lines(x.n,target(x.n,a=2.6,b=6.3),
      col = "blue")
points(v[u<target(v,2.6,6.3)/M(v)],M(x)*u[u<target(v,2.6,6.3)/M(v)],
       col = "blue",
       pch = 16)
points(v[u>=target(v,2.6,6.3)/M(v)],M(x)*u[u>=target(v,2.6,6.3)/M(v)],
       col = "red",
       pch = 4)
legend('topright',
       horiz=F,
       legend = c("candidate","envelope","target","accepted","rejected"),
       col = c("red", "red","blue","blue","red"),
       lty = c("dashed","solid","solid",NA,NA),
       pch = c(NA,NA,NA,16,4)
       )
```
<p>&nbsp;</p> 

![image](https://github.com/user-attachments/assets/ca30f0d4-39e2-4edd-b4f7-d16920a19024) 
<div align="center">
 <i><b>Figure 2</b>: Uniform target candidate in R, n = 10,000.</i>
</div>
<p>&nbsp;</p> 
⬜︎


#### Example 2
Now suppose $X$ is distributed as before, but $V \sim \text{beta}(2,6)$. Recall that we can use inverse transform sampling when $V$ is beta distributed with integer parameters. In particular, if each $U_j$ is an iid standard uniform random variable, then 
```math
$$
V=\frac{\log\left( \prod_{j=1}^{a} U_{j} \right)}{\log\left( \prod_{j=1}^{a+b} U_{j}\right)}=\frac{\log\left( \prod_{j=1}^{2} U_{j} \right)}{\log\left( \prod_{j=1}^{8} U_{j}\right)}
$$
```
will be $\text{beta}(2,6)$ distributed. Thus, while Desmos has no native PRNG for beta distributed variates, we can build one by defining eight uniform random variates (as we defined `U` and `V` in the previous example) and then use the above inverse transform to build `V`.
<p>&nbsp;</p> 

![rejection_sampling_beta_by_beta](https://github.com/user-attachments/assets/a84d77f3-dde2-400e-89e9-df9df61060e1)
<div align="center">
 <i><b>Figure 3</b>: Beta(2,6) target candidate in R, n = 100.</i>
</div>
<p>&nbsp;</p> 

The procedure for visualizing rejection sampling in R is 
```
set.seed(42)
u <- runif(10000,0,1)
v <- rbeta(10000,2,6)

target <- function(x,a,b){
  ((x^(a-1))*((1-x)^(b-1)))/beta(a,b)
}

candidate <- function(x){
  dbeta(x,2,6)
}

M <- (beta(2,6)/beta(2.7,6.3))*(0.7**0.7)*(0.3**0.3)

envelope <- function(x){
  M*candidate(x)
}

#for the condition
M.u.g <- function(x){
  u*envelope(x)
}

plot(x.n,candidate(x.n),
     col="red",
     lty = "dashed",
     type='l',
     ylim = c(min(envelope(x.n)),max(envelope(x.n))),
     xlab = "v",
     ylab = "Mg(v)u"
)
lines(x.n,envelope(x.n),
      col="red", 
      ylim = c(0,4))
lines(x.n,target(x.n,a=2.6,b=6.3),
      col = "blue")
points(v[u<target(v,2.6,6.3)/envelope(v)],M.u.g(v)[u<target(v,2.6,6.3)/envelope(v)],
       col = "blue",
       pch = 16)
points(v[u>=target(v,2.6,6.3)/envelope(v)],M.u.g(v)[u>=target(v,2.6,6.3)/envelope(v)],
       col = "red",
       pch = 4)
legend('topright',
       horiz=F,
       legend = c("candidate","envelope","target","accepted","rejected"),
       col = c("red", "red","blue","blue","red"),
       lty = c("dashed","solid","solid",NA,NA),
       pch = c(NA,NA,NA,16,4)
)
```
<p>&nbsp;</p> 

![image](https://github.com/user-attachments/assets/37d6908f-5f45-4736-ba2d-96707e12ea1b)
<div align="center">
 <i><b>Figure 4</b>: Beta(2,6) target candidate in R, n = 10,000.</i>
</div>
<p>&nbsp;</p> 

⬜︎

Generally, the empirical probability that we accept a sample should be higher in Example 2 than in Example 1, since the candidate distribution is closer to target distribution in the former case.

#### Example 3
Suppose

## Simulation Links (Desmos)
1. [Generating a Beta(2.7,6.3) sample from a Uniform(0,1) sample](https://www.desmos.com/calculator/fvebfd9nzc)
2. [Generating a Beta(2.7,6.3) sample from a Beta(2,6) sample](https://www.desmos.com/calculator/wioa3hc5ry)
3. [Generating a Symmetric Truncated Normal(2,1) sample from a Uniform(0,1) sample](https://www.desmos.com/calculator/ku3vjyysj0)

## References
Casella, G., and Berger, R.L. (2001). *Statistical inference* [2nd ed.]. Cengage.

[^1]: Generally, one should only use a PRNG with a known generation algorithm. For example, the R function `runif()` uses a Mersenne Twister to generate variates, which suffices for most non-cryptographic purposes. However, Desmos is not transparent about the method used to compute random variates.


