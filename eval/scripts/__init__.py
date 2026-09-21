"""Runnable experiments: one module per report, each writing into ``report/``.

A script here is the thing a report is reproducible *from* -- ``python -m eval.scripts.<name>``,
reading its queries out of ``eval/datasets/`` and naming its output through
:func:`config.report_path`, so the config hash in the filename is the hash of the settings the
numbers came from. Nothing in here is imported by the agent or the retrieval layer; the arrows all
point this way.
"""
