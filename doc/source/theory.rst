=====================
Background and Theory
=====================

This page aims to quickly cover some basic survival analysis.

.. note::

    This page is under construction.


Survival Analysis Setup
-----------------------

Suppose we are interested in a particular time-to-event distribution (e.g.,
time until death, time until disease occurrence or recovery, or time until
failure in a mechanical system).
The phrase "time-to-event" can mean nearly any positive quantity being measured,
not necessarily time.

The time-to-event distribution is completely determined by its
*survival function* :math:`S(t) = \Pr(X > t)`, where :math:`X` is the
time-to-event (a positive random variable), and :math:`t` is a positive time.
The survival function might be of interest itself since it answers questions
like, "what is the probability that a cancer patient will survive at least five
more years?" or "what is the probability that this machine part won't break in
the next six months?"

Suppose we have :math:`n` individuals with independent and identically
distributed event :math:`X_1, \ldots, X_n` with survival function :math:`S`.
If we could observe :math:`X_1, \ldots, X_n`, then one obvious candidate for
estimating :math:`S(t)` is the *empirical survival function*:

.. math::

    \widehat{S}(t) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{\{X_i > t\}}.

This is just the number of individuals observed to survive past time :math:`t`
divided by the sample size :math:`n`.
(Here :math:`\mathbf{1}_A` denotes the indicator of an event :math:`A`.)
This unbiased, consistent, and asymptotically normal estimator is nevertheless
not suitable for many situations of interest in survival analysis.
The main problem is that it relies on all the event times :math:`X_1,\ldots,X_n`
being observable.

Censoring and Truncation
~~~~~~~~~~~~~~~~~~~~~~~~

It is common in practice that not all the times :math:`X_1, \ldots, X_n` are
observed.
We list some possible examples.

* A clinical trial investigating a disease treatment might end before a patient
  recovers from the disease, in which case the patient's true recovery time is
  not known.

* A patient whose time until death from cancer is being monitored might die from
  a different disease, so the patient's death-from-cancer time will not be
  known.

* In engineering, a reliability experiment might be stopped after a
  predetermined number of parts fail.
  A part that is still operational after this time will not have its failure
  time observed.

In these cases, the true time-to-event is not known, but a lower bound for it is
available.
This situation is called *right-censoring*.

Another source of incomplete information is *left-truncation* (also known as
*delayed entry*), in which an individual's time-to-event may only be observed if
it exceeds a certain time.
Some examples:

* Patients for a certain disease might only be observed after the disease has
  been diagnosed.
  A patient who died from the disease before diagnosis is unknown to
  investigators.

* If we are observing the age at death of residents in a retirement home, then
  we do not observe the ages at death of individuals who died before becoming
  residents.

* If we are measuring the diameters of particles with a microscope, then only
  particles large enough to be detected by the microscope will be observed.

There are other types of censoring and truncation, but for now we will focus on
right-censoring and left-truncation, the most common variants.

When there is right-censoring or left-truncation, the empirical survival
function is not a suitable estimator of the survival function :math:`S` because
the event times :math:`X_1, \ldots, X_n` are not known.
A popular alternative is the *Kaplan-Meier estimator*, which we will define
below.

Let us first formalize the sampling situation with right-censoring and
left-truncation.
Suppose as before that we have :math:`n` individuals with with independent and
identically distributed times-to-event :math:`X_1, \ldots, X_n` with survival
function :math:`S`.
Moreover, suppose we observe a sample
:math:`(L_1, X_1^\prime, \delta_1), \ldots, (L_n, X_n^\prime, \delta_n)`, where
:math:`L_i` is the *entry time* of the :math:`i`-th individual,
:math:`X_i^\prime` is the observed final time for the :math:`i`-th individual
(so :math:`X_i^\prime > L_i`), and :math:`\delta_i` is zero or one according to
whether :math:`X_i^\prime < X_i` (the :math:`i`-th individual is censored) or
:math:`X_i^\prime = X_i` (the :math:`i`-th individual's time-to-event is
observed).

The Counting Process Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define two stochastic processes associated with this sample: the
*event counting process*

.. math::

    N(t) = \sum_{i=1}^n \mathbf{1}_{\{X_i^\prime \leq t, \delta_i = 1\}},

and the *at-risk process*

.. math::

    Y(t) = \sum_{i=1}^n \mathbf{1}_{\{L_i < t \leq X_i^\prime\}}.

The event counting process counts the number of events that have been observed
to occur up to a certain time, and the at-risk process counts the number of
individuals who are "at risk" (those who are already being observed but have not
yet been censored or experienced the event of interest) at a certain time.

The trajectories (i.e., sample paths) of the event counting process are
right-continuous with left limits (commonly called *càdlàg*). In fact, the event
counting process is non-decreasing and piecewise constant with at most
:math:`n` jumps (the jumps happen at observed event times).
Let :math:`0 < T_1 < T_2 < \cdots` denote the ordered jump times.
The size of the jump at time :math:`T_j` is

.. math::

    \Delta N(T_j) = N(T_j) - N(T_j-),

where for :math:`t > 0`,

.. math::

    N(t-) = \lim_{s \uparrow t} N(s)

is a limit from the left.
In other words, :math:`\Delta N(T_j)` is the number of observed events at time
:math:`T_j`.
For convenience, also define :math:`T_0 = 0`.

Stochastic integrals with respect to :math:`N` can be easily computed as sums:

.. math::

    \int_0^t H(s) \, dN(s) = \sum_{j : T_j \leq t} H(T_j) \Delta N(T_j).

The Kaplan-Meier Estimator
--------------------------

We will now give a derivation of the Kaplan-Meier survival function estimator.
This will be an estimator of the survival function :math:`S(t) = \Pr(X > t)`,
where :math:`X` is the time-to-event, based on the right-censored and
left-truncated sample
:math:`(L_1, X_1^\prime, \delta_1), \ldots, (L_n, X_n^\prime, \delta_n)`
described above.

First, observe that if we have times :math:`s < t`, then

.. math::

    S(t)
    &= \Pr(X > t) \\
    &= \Pr(X > s) \Pr(X > t \mid X > s) \\
    &= S(s) \Pr(X > t \mid X > s).

Thus, it suffices to estimate the conditional probability
:math:`\lambda(s, t) = \Pr(X > t \mid X > s)` of surviving the time interval
:math:`(s, t]` given survival up to time :math:`s`.


.. todo::

    Finish this section.
