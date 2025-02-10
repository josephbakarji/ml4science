### Everything is a Function

I have a story that I like to tell - one that begins with the claim that
everything is made of functions. In other words, much of what we see in the
world can be thought of as inputs and outputs governed by some mapping. I have a
personal hypothesis — purely speculative at this stage — that the main job of
the human brain is to learn precisely such a function: a mapping between actions
we take and the perceptions we receive.

When I consider how the brain might represent something that cannot be acted
upon directly, I see it creating a surrogate or internal model — a
representation — that can be manipulated in place of the real thing. It can even
nest representations within representations, all of which are functions, or
transformations, of some inputs into some outputs. The catch, of course, is that
these transformations in our brains unfold in time. We send a signal now and
receive feedback later, so each function spans at least two points in time, not
just one.

We can push this concept further by noting that we never really occupy the same
point in space for long, and time never stands still. Quantum mechanics
introduced us to the idea that even "fixed" points in space are not so fixed
after all. Everything moves, and in that sense, there is a non-local flavor to
the most ordinary exchanges of signals: a function we enact here and now might
only reveal its consequences somewhere else, at some future moment.

Yet from these puzzling pieces, we build laws that let us predict the future
with surprising success. We stitch partial derivatives, algebraic constraints,
and boundary conditions together almost like a set of Tetris blocks. Combine
enough of these fragments, and suddenly we gain the power to forecast what had
once been uncertain. It reminds me of how we say "a new law was derived": we
start with known relationships, place them side by side, and arrive at an
insight that generalizes beyond our original data. This act of generalization is
ancient—predating modern science — because all human language, thought, and
code-making depend on the ability to convey something that hasn’t yet been
encountered.

From this perspective, machine learning is yet another tool for fitting these
functions. It can map from images to text, or from sequences of words to the
next word in a sentence, echoing what the human brain does when it forms
language. We measure relationships in data and train models to capture them. The
challenge, of course, is that we might only have limited data. Fitting a
function too tightly to the things we already know is overfitting—it fails to
generalize. And what we prize, in both science and engineering, is the capacity
to predict something new.

One of my goals in teaching this material is to highlight that prediction is
the real test of our laws and models. In purely practical terms, you can have a
chatbot that tells you how to fix your code, but for truly scientific insights —
especially those that stretch beyond known facts — we need the rigor of
modeling, testing, and exploring the limits of our assumptions. Generating text
or code might look impressive, but the deeper question is whether these systems
can support genuine discoveries.

I plan to illustrate this with a simple homework exercise. I’ll give out an
equation, have you solve it to get a time - dependent solution, and then ask you
to identify the original equation just by looking at the resulting data. It
might sound like a pointless exercise at first — after all, in that scenario we
already know the equation that generated the data — but it's an excellent
training ground for the real problem. In reality, we often have data from some
unknown process, and we'd like to reverse-engineer the underlying rules.

I find it fascinating to watch people try to reconstruct a known law from data
they generated themselves. Sometimes, it goes surprisingly well; other times,
unforeseen subtleties emerge, and the reconstruction fails. Such challenges
mimic real research scenarios, where we measure phenomena, guess at the
underlying mechanisms, and refine our guesses until we land on something that
predicts future measurements.

In many contexts, we also care about actions and outcomes. The brain, for
instance, seems to take actions and observe the consequences, constantly
refining its internal model of how the world responds. The field of embodied
intelligence explores precisely this feedback loop. It raises the question: how
important is action in developing an understanding of objects and processes? We
know that simply observing data can be quite powerful, but active
experimentation — poking and prodding the system — often reveals hidden
structure more quickly.

Thinking along these lines, I notice that much of what we call "intelligence",
whether human or machine, remains an open question. Even now, there's no perfect
consensus on what science actually is at a fundamental level. We practice it
daily, produce vast numbers of scientific papers, and build models that work
well enough to fly airplanes and predict the weather for a few days. But the
philosophical underpinnings — why our models ever work at all and when they're
expected to fail — are still a little mysterious. A problem philosophers have
been wrestling with [since the birth of
science](https://en.wikipedia.org/wiki/Problem_of_induction)

This leads me to worry about the dangers of overfitting in a societal or
intellectual sense. When we overfit to what we already know, we risk stifling
creativity and innovation. Sometimes, the most significant breakthroughs come
from discarding old assumptions or stepping into territory that initially seemed
illogical or useless. This is why I believe that creativity is absolutely
essential; especially in science and engineering. It may not take the same form
as painting a masterpiece, but it involves the same willingness to experiment,
fail, try again, and occasionally break from tradition.

I see this process as a constant interplay between what is known and what could
be newly discovered. If we rely too heavily on a single, highly-refined model of
the world, we might never question its core assumptions or notice the anomalies
it fails to explain. George Box famously said, “All models are wrong, but some
are useful.” I agree, though I also notice that any blanket statement about all
models is itself a kind of model — an idea that underscores how recursive and
paradoxical this realm of meta-modeling can get. ([Reminiscent of Godel's
incompleteness
theorems](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems))

The practical upshot, for anyone seeking to automate or augment the scientific
process, is that we have to be comfortable stepping beyond polished solutions
and consider new approaches, new mathematics, and new ways of seeing. We must
get our hands dirty with data—looking at it in varied ways, noticing patterns,
and pushing ourselves to hypothesize equations or algorithms that capture those
patterns.

When it comes to data-driven modeling, there is hope that deep learning assumes
as little about the models as possible and purely learns the patterns in the
data. However, current deep learning approaches are reflex models only capture
one level of abstraction; unlike the human scientific process, which is capable
to build models upon models, dismanteling them and rebuilding them meticulously.
As long as we don't understand what that process truly entails, both on the data
collection and modeling side, automating the scientific process will remain
illusive.

In a few weeks, you will tackle a competitive challenge. I’ll share time-series
data generated by a known (to me) differential equation, and everyone else will
try to infer that equation from the data alone. It should be eye-opening to see
which strategies actually capture the underlying dynamics and which simply
memorize noise. Some might find the exact differential operator, others might
approximate it in a way that extrapolates well with a black box model.
Ultimately, this is the spirit of real discovery: looking for a representation
that both fits what we see and predicts what we haven’t yet seen.

Amid all these details, one of the main lessons I want to convey is that time
and history matter immensely. A system might depend not just on its immediate
state but on an entire history of previous states. Or it might skip and only
depend on certain delayed points in time. We can encode such dependencies in
integral-differential equations, delay differential equations, or memory-based
models. Materials like viscoelastic fluids really do "remember" how they’ve been
deformed in the past. This opens a vast zoo of possible analogies and modeling
techniques, many of which historically were tackled without computers by
combining artistry with mathematics.

Now, of course, we can rely on large-scale computing to approximate nearly any
function we can dream up, which is both exciting and dangerous. Exciting,
because we can tackle problems too complex to solve by pencil and paper.
Dangerous, because it's easy to lose intuition and latch onto solutions that
match our existing expectations, inadvertently missing the real, messy story.
Some phenomena — like quantum mechanics — remain linear at their core and yet
still produce emergent chaotic behaviors when coupled to larger systems. It's a
humbling reminder that even "simple" rules can lead to deep complexities.

Ultimately, what I hope people take away from this approach is a sense of
exploration and respect for the unpredictability at the heart of so many
processes. I don't expect that knowledge of overfitting or chaos will radically
change anyone's daily routine, but it might instill a healthy skepticism
whenever we assume we really know how something works. If nothing else, stepping
into this mindset can help us see new angles, gather the kind of data we
otherwise might neglect, and dream up fresh ideas that push our field forward.
