"""Dataset report generation (HTML, PDF, CSV, LaTeX tables).

The implementation lives in the ``AaltoAD.report`` submodules: ``cli`` (option
vocabulary and input normalization), ``metrics`` (metric lookup and
comparison), ``plot`` (prediction-error plots) and ``report`` (tables, output
formats, orchestration). ``generate_report`` takes the parsed CLI namespace;
tests import the implementation modules directly.
"""

from AaltoAD.report.report import generate_report
