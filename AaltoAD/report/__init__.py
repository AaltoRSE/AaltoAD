"""Dataset report generation (HTML, PDF, CSV, LaTeX tables).

The implementation lives in ``AaltoAD.report.report`` (moved here from
``AaltoAD/report.py``). Only the names used by other production code are
re-exported here; tests import the implementation module directly.
"""

from AaltoAD.report.report import generate_report, method_name

