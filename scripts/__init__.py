"""Repo-maintenance tools that are not part of the library and are never imported by it.

``eval/scripts/`` holds the experiments -- things that read the corpus and write a report.
This package holds the ones that maintain the repository itself: today only
:mod:`scripts.export_requirements`, which regenerates the pip requirements files from
``uv.lock``. The other half of AlmaBot (``almabot-web``) uses the same name for the same kind
of thing, its ``scripts/build_data.py``.
"""
