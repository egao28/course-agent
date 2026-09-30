"""The evaluation half: the dataset's contract, the metrics, and the two fitted thresholds.

:mod:`eval.schema` is the six-field row that ``eval/datasets/v1.jsonl`` commits to and
:mod:`eval.validate` is the runnable check over a file of them; :mod:`eval.metrics` holds
recall@k and MRR@k, hand-written and short enough to check by eye; :mod:`eval.conformal` fits
RQ3's answering threshold on the calib half and :mod:`eval.intents` carries the hand-labelled gold
intents for the test half.

The runnable experiments live one level down in :mod:`eval.scripts`, which writes into ``report/``.
Nothing in either package is imported by :mod:`agent` or :mod:`retrieval`; the arrows all point
this way, so an eval can read the system without the system being able to read the eval.
"""
