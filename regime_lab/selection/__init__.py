"""Regime as a contextual selector across a signal library — the Two Sigma plan.

`pilotage/plans_de_recherche/twosigma/PRESPEC_TWOSIGMA.md` is the specification. It
is a DRAFT: nothing in this package reads the tree's answer, and nothing here settles
a choice the draft leaves open. Each module names the choices it found.

``library``   the twelve candidates, the two exclusions, the ten-signal ``LIBRARY``,
              and position-space diagnostics (geometry, turnover).
``folds``     the walk-forward: five test folds of 4.0 years on the inferential sample.
``context``   the context partition, its walk-forward refit, out-of-sample labels and
              each fold's own label path on the industry sessions.
``protocol``  how signals become one book: blend, tilt, volatility target, costs,
              excess returns, and the blinded power convention.

The matched placebo lives in ``regime_lab.analysis.placebo``, shared by the programme.
"""
