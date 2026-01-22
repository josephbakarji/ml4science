---
title: "The Gauss-Ceres Story: Birth of Least Squares"
layout: note
permalink: /static_files/lectures/02/notes_gauss_ceres/
---


## Historical Context

### The Missing Planet Problem (Titius-Bode Law)

In 1766, Johann Titius noticed a mathematical pattern in planetary distances from the Sun:

$$D(n) = \frac{3 \times 2^n + 4}{10} \text{ AU}$$

| n | Predicted (AU) | Planet | Actual (AU) |
|---|----------------|--------|-------------|
| -∞ | 0.4 | Mercury | 0.39 |
| 0 | 0.7 | Venus | 0.72 |
| 1 | 1.0 | Earth | 1.00 |
| 2 | 1.6 | Mars | 1.52 |
| 3 | **2.8** | **???** | — |
| 4 | 5.2 | Jupiter | 5.20 |
| 5 | 10.0 | Saturn | 9.55 |

The formula predicted a planet at 2.8 AU—but nothing was there. Astronomers believed a "missing planet" existed in the gap between Mars and Jupiter.

When Uranus was discovered in 1781 at 19.2 AU (predicted: 19.6), the law gained credibility, and the hunt for the missing planet intensified.

### The Discovery of Ceres (January 1, 1801)

Giuseppe Piazzi, observing from the Palermo Observatory in Sicily, spotted a faint object moving against the background stars. He tracked it for **41 days** through only **9 degrees** of arc before it disappeared into the Sun's glare.

**The problem**: With less than 1% of its orbit observed, how could astronomers find it again when it emerged months later?

## The Mathematical Challenge

### What Piazzi Had

- **22 observations** over 41 days
- Each observation: a timestamp + 2 angles (right ascension, declination)
- That's 44 data points total

### What Needed to Be Found

An elliptical orbit requires **6 parameters** (orbital elements):

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Semi-major axis | a | Size of the ellipse |
| Eccentricity | e | Shape (0 = circle, approaching 1 = elongated) |
| Inclination | i | Tilt of orbital plane relative to ecliptic |
| Longitude of ascending node | Ω | Where orbit crosses ecliptic (ascending) |
| Argument of perihelion | ω | Orientation of ellipse within orbital plane |
| Time of perihelion | T₀ | When the object is closest to the Sun |

### Why It Seemed Impossible

- Previous methods assumed either circular orbits (planets) or parabolic orbits (comets)
- Ceres's eccentricity was unknown—could be anything between 0 and 1
- The great mathematician Laplace declared the problem **unsolvable** with so little data

## Gauss's Solution

### The 24-Year-Old Genius

Carl Friedrich Gauss, already known for his mathematical brilliance, took on the challenge. He had secretly developed a new technique years earlier but never published it.

### His Approach

1. **No assumptions about eccentricity**: Unlike others, Gauss didn't guess whether the orbit was circular or elongated

2. **Selected 3 well-spaced observations**: January 1, January 21, and February 11
   - Three observations give 6 angles
   - 6 angles can determine 6 orbital parameters (in principle)

3. **Iterative refinement**: Used all 22 observations to improve the estimate

4. **The key insight**: Find parameters that minimize the **sum of squared errors** across ALL observations, not parameters that fit any single observation exactly

### Three Months of Calculation

Gauss performed over 100 hours of hand calculations. He even invented early versions of what we now call the Fast Fourier Transform to speed up his computations.

## The Triumph

In December 1801, Gauss published his predicted position for Ceres.

**The surprise**: His prediction was **6 degrees away** from where other astronomers expected it to be.

**December 31, 1801**: Franz Xaver von Zach pointed his telescope at Gauss's predicted location and found Ceres—**within half a degree** of the prediction.

Von Zach wrote: *"Ceres is now easy to find and can never again be lost, since the ellipse of Dr. Gauss agrees so exactly with its location."*

## The Method of Least Squares

### The Core Idea

When you have noisy measurements and want to fit a model:

- You **cannot** pass through every data point exactly
- Instead, find parameters that minimize the **total error**
- Specifically: minimize the **sum of squared errors**

$$E = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where $y_i$ is the observed value and $\hat{y}_i$ is the predicted value.

### Why Squared?

1. **Positivity**: Errors can be positive or negative; squaring makes them all positive
2. **Symmetry**: Overestimates and underestimates are treated equally
3. **Large error penalty**: Squaring penalizes large errors more than small ones
4. **Mathematical convenience**: The derivative of $e^2$ is simply $2e$
5. **Statistical optimality**: If errors are normally distributed, least squares gives the maximum likelihood estimate

### Publication and Priority

- Gauss claimed he developed the method in 1795 (age 18)
- He didn't publish until 1809 in *Theoria Motus Corporum Coelestium*
- Legendre independently published the method in 1805
- This led to a priority dispute, but Gauss's contribution was deeper: he connected least squares to probability theory and the normal distribution

## Legacy

The method of least squares remains fundamental to:

- Astronomy and orbital mechanics
- Geodesy and surveying
- Statistical regression
- Machine learning
- Signal processing
- Any field where we fit models to noisy data

Today, we still use Gauss's methods to:
- Track asteroids and predict potential Earth impacts
- Fit models in machine learning
- Process GPS signals
- Analyze experimental data

## References

- [Gauss and Ceres - Rutgers Math History](https://sites.math.rutgers.edu/~cherlin/History/Papers1999/weiss.html)
- [Gauss, Least Squares, and the Missing Planet - Actuaries Institute](https://www.actuaries.asn.au/research-analysis/gauss-least-squares-and-the-missing-planet)
- [Gauss Predicts the Orbit of Ceres - ThatsMaths](https://thatsmaths.com/2021/06/24/gauss-predicts-the-orbit-of-ceres/)
- [Wikipedia: Ceres (dwarf planet)](https://en.wikipedia.org/wiki/Ceres_(dwarf_planet))
- [Wikipedia: Titius-Bode Law](https://en.wikipedia.org/wiki/Titius–Bode_law)
