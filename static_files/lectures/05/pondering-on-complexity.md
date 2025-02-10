### Pondering on the Nature of Complexity 

I have always found it both remarkable and a little mysterious that
mathematics, an abstract construct of symbols and rules, can capture so many of
the phenomena we see around us. As a student, I distinctly remember the moment I
realized that algebra and calculus do more than just rearrange numbers—they
actually describe the motion of planets, the flow of fluids, the vibrations of
strings. This was my first real encounter with the power of math to explain the
natural world rather than merely calculate within it.

Over time, I became increasingly curious about the nature of scientific
modeling. We often speak of equations or simulations as if they are the reality
itself, yet I see them more as analogies—compressed representations of
observations and insights. Any mathematical framework, from differential
equations to neural networks, attempts to emulate how the world behaves without
being the world itself. Indeed, although we treat these models as if they were
true, each one is ultimately a metaphor, a stand-in that helps us reason about
physical processes. Or as George Box, the statistician, would express it:

    All models are wrong, but some are useful.

While this saying is a little paradoxical (because it is itself a model about
models), it has shaped my perspective on both science and machine learning.
There is always a gap between the real world and the model we use to describe
it. At some point that gap is too big for the whole modeling framework to be
useful. 

Whenever I teach courses on the subject, I notice students are focused on
picking up the coding and the optimization routines. However, I usually want
them to see the bigger picture: that neural networks aren’t just clever
algorithms. They hint at something deeper about how systems learn, adapt, and
generate structure from raw information. On some level, that is exactly what we
do as human beings when we observe the world and form abstract concepts. That
the process of scientific discovery, that of training neural networks, and that
of neural networks themselves, are all nonlinear and unpredictable. That’s what
makes all these endeavors at least partially art-forms.

It’s easy to forget that our entire approach to science relies on simplifying
assumptions. For instance, we typically emphasize linear systems in engineering,
because linearity is much more tractable. If a system is linear, small changes
in one component will stay small, and linear superposition lets us predict
outcomes by straightforward addition of effects. But the moment we allow for
nonlinearity, we invite phenomena that can quickly get out of hand: chaos,
turbulence, tipping points. Nature is deeply nonlinear, and so we encounter
these complexities in everything from the weather to fluid flow to neural
networks.

As a postdoc at the University of Washington, I spent a lot of time in a
research group that was obsessed with such nonlinear systems. Particularly, the
famous Lorenz equations which is a reduced order model that models the behavior
of weather patterns, using three differential equations: 

$$
\begin{align*}
\frac{dx}{dt} &= \sigma(y-x) \\
\frac{dy}{dt} &= x(\rho-z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{align*}
$$

where $$\sigma$$, $$\rho$$, and $$\beta$$ are parameters that determine the
system's behavior (e.g. $$\sigma$$ is the Prandtl number, $$\rho$$ is the
Rayleigh number, and $$\beta$$ is the ratio of the fluid's thermal expansivity
to its compressibility). The classic values that Lorenz used are $$\sigma=10$$,
$$\rho=28$$, and $$\beta=8/3$$, which produce the famous butterfly-shaped
strange attractor.

![Lorenz Attractor](https://images.app.goo.gl/ufGj9qJPkyQRtesaA)

Those three deceptively simple differential equations were originally meant to
capture some rudimentary aspects of atmospheric convection. Yet they revealed a
confounding truth: deterministic systems can be fundamentally unpredictable over
longer time horizons. If you start with slightly different initial conditions,
the solutions diverge exponentially, making it impossible to forecast the
system’s future beyond a certain point. Although we might know the exact rules
governing the dynamics, our finite precision measurements ensure we lose track
of what will happen eventually. That phenomenon still fascinates me, because it
highlights a stark difference between writing down a neat equation and actually
knowing how a system will behave.

Curiously, neural networks exhibit their own sort of complexity. Instead of
physically swirling fluids, we have digital neurons (essentially little
nonlinear functions) adjusting their parameters in response to data. Each
forward pass applies the current parameters to the input, and each backward pass
fine-tunes those parameters based on the error. The network organizes itself
from the bottom up, discovering features we never explicitly programmed. This is
how countless processes in nature find structure: cells grow according to local
signals, storms form through subtle interactions of pressure differences,
societies evolve through individuals’ small actions converging (or diverging) en
masse.

Yet our engineering mindset has often been to avoid such bottom-up complexity
precisely because it’s so difficult to control. We build bridges, circuit
boards, and even software systems with linear assumptions, decoupled submodules,
and predictable interactions. While that approach has delivered enormous
technological advances, we risk missing out on the richness that arises from
nonlinearity and self-organization. When we thought that we can program
computers to be intelligent using hard-coded rules, we missed out on the
complexity that can emerge by just letting computers figure out what are the
rules from experience.

So don’t treat machine learning purely as a set of recipes: proposing a
hypothesis (or neural network architecture), optimizing, testing, etc. Start by
posing genuine questions about the world whose answers we do not already know;
and preferably one you’re excited to answer. If our only aim is to pick an
architecture and tune the hyperparameters until we get a good accuracy, we lose
sight of the open-ended nature of science and ML alike. To me, machine learning
becomes truly powerful when it is a lens through which we can peer into the
underlying complexity of phenomena, whether natural or synthetic, and uncover a
pattern or mechanism we didn’t see before.

It’s also good to remember that our scientific tools are limited reflections of
infinite realities. In analogy, real numbers, for instance, carry an infinity of
decimal places, yet our computers and measurement devices can only store or
observe a finite amount. In chaotic systems, that finite difference expands
exponentially over time, leading to fundamentally different predictions. So we
should remain humble in the face of nonlinear dynamics and remember that
although an equation or model might look simple, it can give rise to immense
complexity.

All of these considerations shape the stories I’m trying to tell in this
course. I often weave stories together - accounts of historical breakthroughs,
personal experiences grappling with equations, and glimpses of modern machine
learning research - so that everyone can get a feel for what doing science
genuinely entails. At its core, science is an adventure in modeling and
discovery. Differential equations, neural networks, chaos theory - these are not
separate silos; they converge on the question of how to make sense of a world
where small tweaks can cause big upheavals and where emergent order can arise
from seemingly disordered interactions.

I plan to continue sharing these perspectives in class because I believe they
offer a richer context for why we do what we do in scientific machine learning.
And I invite you to help shape this exploration. Perhaps we can organize
hackathons, collaborative projects, or deep-dive discussions into particular
phenomena. When we co-create the course’s direction, we capture some of the very
essence of research: building on each other’s curiosity and stumbling upon
insights that none of us would have found alone.