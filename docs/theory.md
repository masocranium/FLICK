# Theory

How FLICK's default model works, and why each piece exists. It assumes you're
comfortable with probability and linear algebra, but not that you've met copulas
before. For other models, see [models.md](models.md). For trajectories, see
[paths.md](paths.md).

## The problem

We have a table of $n$ rows and $p$ columns, where $n$ is small (often 100 to 500)
and $p$ is around 14. We want to generate new rows that have the right distribution
for each column and the right dependence between columns.

Two simple models fail:

- **Independent marginals** get each column right and lose all dependence.
- **A multivariate Gaussian** keeps linear dependence but forces every marginal to
  be Gaussian. That produces negative durations and fractions above 1, and it can't
  be fitted at all when the sample covariance is singular.

## Copulas

Sklar's theorem says any joint distribution can be split into its marginals $F_j$
and a copula $C$, a distribution on the unit cube that holds only the dependence:

$$F(x_1, \dots, x_p) = C\big(F_1(x_1), \dots, F_p(x_p)\big)$$

So the two problems can be solved separately. This package uses the **Gaussian
copula**, the dependence structure of a multivariate normal with correlation matrix
$R$. A Gaussian copula with Gaussian marginals *is* a multivariate Gaussian, so this
strictly generalizes that model.

## Sampling: NORTA

NORTA ("NORmal To Anything", Cario and Nelson, 1997) generates a row in three steps:

1. Draw $Z \sim \mathcal{N}(0, R)$, using the Cholesky factor of $R$.
2. Map each coordinate to a uniform: $U_j = \Phi(Z_j)$.
3. Push each through its marginal's inverse CDF: $X_j = F_j^{-1}(U_j)$.

The output has exactly the marginals $F_j$. Its correlations are close to $R$, but
not generally equal to it, and that gap is the
[correlation-matching problem](#norta-correlation-matching).

## Marginals

`choose_marginal` tests a column in this order. The order matters:

1. **Constant**, if the column has no variation. This has to come before anything
   that divides by a spread.
2. **Discrete**, if the column is integer-valued with at most 12 distinct values.
   This comes before the atom test so a count's most common value isn't mistaken
   for a point mass.
3. **Hurdle**, if one value repeats at least 3 times and holds at least 5% of the
   mass. Zero is tested first and with a tolerance, because derived columns hold
   floating-point dust such as `0.0`, `-0.0` and `7e-15`, which exact equality
   would miss.
4. **KDE**, for everything else.

The repeat requirement matters at small $n$. In a 17-row table a single value is
5.9% of the data, so the 5% share alone would call any value an atom.

### KDE in a transformed space

A plain Gaussian KDE has unbounded support, so it would emit impossible values too.
The KDE is therefore fitted after a transform chosen from the column's declared
bounds:

| Bounds | Transform | Inverse |
|---|---|---|
| None | $z = x$ | $x = z$ |
| $(l, \infty)$ | $z = \log(x - l)$ | $x = l + e^{z}$ |
| $(-\infty, h)$ | $z = -\log(h - x)$ | $x = h - e^{-z}$ |
| $(l, h)$ | $z = \operatorname{logit}\frac{x - l}{h - l}$ | $x = l + (h - l)\,\sigma(z)$ |

Every inverse maps the whole real line into the allowed range, so impossible values
can't be reached.

The negative sign on the upper-bounded transform is essential. Without it the
transform would reverse order, and the resulting "CDF" would run from 1 down to 0.
Marginals would still sample correctly, which hides the error, but every latent
correlation involving that column would flip sign.

The bandwidth is Silverman's rule, $h = 0.9 \min(s, \mathrm{IQR}/1.349)\, n^{-1/5}$,
computed on the transformed values, where the rule's near-Gaussian assumption is
more reasonable.

The CDF is computed once onto a 2048-point grid, and both `cdf` and `ppf` interpolate
that grid, which keeps them consistent inverses of each other.

A KDE is used rather than an empirical CDF because inverting an empirical CDF can
only return values already observed, which hard-truncates the tail.

### Hurdle marginals

No continuous density can put mass on a single point. A hurdle marginal is a mixture
of a point mass with probability $\pi$ at atom $a$, and a continuous body $G$ fitted
to the remaining values:

$$F(x) = (1 - \pi)\,G(x) + \pi\,\mathbf{1}[x \ge a]$$

## Fitting the correlation

Fitting a model with columns $X$ runs these steps in order:

1. Fit each marginal.
2. Map each column to uniform scores, then to normal scores $Z = \Phi^{-1}(U)$.
3. Take $R$ as the Pearson correlation of $Z$.
4. Calibrate the attenuated pairs.
5. Shrink.
6. Project back to a valid correlation matrix.

### The distributional transform

For a continuous marginal, $U = F(X)$ is uniform. With ties it isn't: every tied
value lands on the same $U$, and Sklar's theorem no longer identifies a unique
copula. The distributional transform (Rüschendorf, 2009) spreads ties across the jump
they share:

$$U = F(x^-) + V\,\big(F(x) - F(x^-)\big), \qquad V \sim \mathrm{Uniform}(0, 1)$$

This restores uniformity, which is why each `Marginal` exposes `cdf_interval`.

### NORTA correlation matching

The distributional transform has a cost: adding independent noise to a variable
biases its correlations toward zero. So the latent $\rho$ you estimate and the $\rho$
you need to reproduce the data can be far apart. In one recorded 182-row table, a
pair came out at $-0.519$ when reproducing its observed Spearman correlation of
$-0.508$ required a latent value of $-0.825$.

For two marginals without atoms, no correction is needed. $F^{-1}$ is strictly
increasing, so it preserves ranks exactly, and the sampled Spearman correlation is a
fixed function of the latent one:

$$\rho_S = \tfrac{6}{\pi} \arcsin(\rho / 2)$$

With atoms there's no closed form, so `calibrate_pair` solves it numerically. For a
proposed $\rho$ it simulates through the fitted marginals, measures the Spearman
correlation, and bisects. The simulation reuses one pair of independent normal
vectors across the whole search:

$$Z_a = Z_0, \qquad Z_b = \rho Z_0 + \sqrt{1 - \rho^2}\,Z_1$$

Those common random numbers keep the $\rho \to \rho_S$ curve smooth, so bisection
converges rather than chasing Monte Carlo jitter. If the target is out of reach (two
atoms can cap the attainable rank correlation well below 1), the nearest end of the
range is returned.

Two choices here differ from the textbook method:

- **It matches Spearman, not Pearson.** Cario and Nelson match Pearson correlation.
  Rank correlation is more robust on the heavy-tailed columns here, and is invariant
  to the marginal transforms.
- **It only calibrates pairs that are badly off.** A pair must touch an atom and
  miss its target by more than `min_attenuation` (default 0.15). Calibrating every
  atom-touching pair makes each one individually right and the matrix jointly
  unrealizable, and the repair then costs more than it gains. On one recorded table,
  held-out energy distance was 0.2997 uncalibrated, 0.3091 with every such pair
  calibrated, and 0.2948 with the threshold.

**A calibrated entry isn't a correlation you can read directly.** It's deliberately
inflated so the *sampled* correlation lands on target. `GaussianCopula` records which
pairs were calibrated, and the CLI marks them.

### Shrinkage

With $p = 14$ there are 91 correlations to estimate, which is a lot from a few
hundred rows. Shrinking toward the identity trades a little bias for much less
variance:

$$\hat R = (1 - \lambda) R + \lambda I$$

The intensity is the Ledoit-Wolf (2004) estimate:

$$\lambda = \frac{\min(\bar b^2, d^2)}{d^2}$$

Here $d^2$ measures how far the sample correlation sits from the identity, and
$\bar b^2$ measures how noisily it's estimated. Two datasets with the same $n$ and
$p$ can deserve very different shrinkage, which a fixed rule such as $p / (p + n)$
can't express. That rule remains as the fallback when there's no latent matrix to
estimate from.

Calibration runs *before* shrinkage because they address different problems:
calibration removes a bias, and shrinkage reduces variance. In the other order,
calibration would undo the shrinkage on exactly the entries it touched.

### Keeping the matrix valid

Pairwise calibration can produce a matrix that isn't positive semidefinite, because
individually correct correlations need not be jointly realizable. This is a known
property of NORTA as dimension grows (Ghosh and Henderson, 2003).

`nearest_correlation` repairs it with Higham's (2002) alternating projections, which
minimize the total Frobenius distance and so spread the repair across every entry
that has to give. The cruder `nearest_psd` (clip eigenvalues, rescale) can load the
whole correction onto a few entries. It's used only when calibration changed nothing
and the matrix is already nearly valid, and as a final eigenvalue floor so Cholesky
can't fail.

## Conditional sampling

Partition the latent vector into known $a$ and unknown $b$. The conditional of a
multivariate normal is available in closed form:

$$Z_b \mid Z_a = z_a \sim \mathcal{N}\big(R_{ba} R_{aa}^{-1} z_a,\; R_{bb} - R_{ba} R_{aa}^{-1} R_{ab}\big)$$

`sample_conditional` maps known values to $z_a$, draws $Z_b$, and pushes those
through their marginals. Known values use the *midpoint* of the CDF interval rather
than a random spread, so conditioning is a deterministic function of what you know:
two calls with the same value and the same RNG state agree.

Be careful conditioning far outside the observed range. The CDF saturates at the
edge of its grid, so a known value beyond that is treated like one at the edge, and
the model won't extrapolate further.

## Numerics

The package has no SciPy dependency. $\Phi$ comes from `math.erf`, and $\Phi^{-1}$
is Acklam's rational approximation, with a relative error of about $1.15 \times
10^{-9}$. Inputs are clipped to $[10^{-12}, 1 - 10^{-12}]$, because a single infinite
latent value would turn the entire correlation estimate into NaN.

## References

- Acklam, P. J. *An algorithm for computing the inverse normal cumulative
  distribution function.* (Web note.)
- Cario, M. C., and Nelson, B. L. (1997). *Modeling and generating random vectors
  with arbitrary marginal distributions and correlation matrix.* Technical report,
  Northwestern University.
- Ghosh, S., and Henderson, S. G. (2003). *Behavior of the NORTA method for correlated
  random vector generation as the dimension increases.* ACM Transactions on Modeling
  and Computer Simulation, 13(3).
- Higham, N. J. (2002). *Computing the nearest correlation matrix: a problem from
  finance.* IMA Journal of Numerical Analysis, 22(3).
- Ledoit, O., and Wolf, M. (2004). *A well-conditioned estimator for large-dimensional
  covariance matrices.* Journal of Multivariate Analysis, 88(2).
- Rüschendorf, L. (2009). *On the distributional transform, Sklar's theorem, and the
  empirical copula process.* Journal of Statistical Planning and Inference, 139(11).
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis.*
  Chapman and Hall.
- Sklar, A. (1959). *Fonctions de répartition à n dimensions et leurs marges.*
  Publications de l'Institut de Statistique de l'Université de Paris, 8.
- Székely, G. J., and Rizzo, M. L. (2013). *Energy statistics: a class of statistics
  based on distances.* Journal of Statistical Planning and Inference, 143(8).
