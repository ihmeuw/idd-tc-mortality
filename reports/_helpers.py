"""DEPRECATED — moved to idd_tc_mortality.viz.screening.

These helpers now live in the package (``from idd_tc_mortality.viz import screening``) so they
import without a sys.path hack. This file is a TEMPORARY re-export shim kept only so existing
notebooks / qmd that do ``import _helpers as H`` keep working during migration; it will be
deleted once callers are repointed. Do not add new code here — put it in viz/screening.py.
"""
from idd_tc_mortality.viz import screening as _screening

globals().update({k: v for k, v in vars(_screening).items() if not k.startswith("__")})
