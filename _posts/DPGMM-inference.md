#Given an existing cluster assignment for $y_1, ... y_n$, the probability of $y_{n+1}$ being assigned to cluster $k$ is $\frac{\|C_k\|}{\alpha+n}$, where $\|C_k\|$ is the number of datapoints assigned to the cluster, and the probability of starting a new cluster: $\frac{\alpha}{\alpha+n}$

#$x_i\|\theta_k \sim F(\theta_k)$ \\
#$x_i\|c_k, \mu_k \sim N(\mu_k)$

#where $c_k$ indicates the cluster associated with observation $x_i$.
#We use a symmetric Dirichlet prior with concentration parameter $\alpha/K$.
