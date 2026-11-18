#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Akshara edits: new module (whole file)

"""
Evolutionary corrections to observed [C/Fe] from Placco et al. (2014, ApJ,
797, 21), via the online calculator at https://vplacco.pythonanywhere.com/.

Based on the approach of the `carbcor` package
(https://github.com/guilimberg/carbcor), reimplemented here without the
pandas/BeautifulSoup dependencies. The calculator corrects observed [C/Fe] for
the depletion of carbon by CN processing on the upper red giant branch.
The inputs are log(g), [Fe/H] and the observed [C/Fe], with uncertainties for a
Monte Carlo estimate of the correction uncertainty (median absolute deviation).
The result is for the [N/Fe] = 0.0 case.

Parameter space (inputs outside it are clipped by the calculator to the
nearest grid point):
    log(g) :  0.0 to 5.0  (0.1 steps)
    [Fe/H] : -5.5 to 0.0  (0.1 steps)
    [C/Fe] : -1.0 to 3.0  (0.1 steps)
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import re
import numpy as np
import requests

PLACCO_URL = "https://vplacco.pythonanywhere.com/"
PLACCO_REFERENCE = "Placco, V.M., Frebel, A., Beers, T.C., & Stancliffe, R.J. 2014, ApJ, 797, 21"
PLACCO_LIMITS = {
    "logg": (0.0, 5.0),
    "[Fe/H]": (-5.5, 0.0),
    "[C/Fe]": (-1.0, 3.0),
}
_NUM = r"[-+]?\d+(?:\.\d*)?"
_RESULT = re.compile(
    r"correction\s+is:\s*(" + _NUM + r")"
    r"(?:\s*(?:&plusmn;|±)\s*(" + _NUM + r")\s*\(MAD\))?"
    r"(?:\s*\(\s*\[N/Fe\]\s*=\s*(" + _NUM + r")\s*case\s*\))?",
    re.IGNORECASE)


def placco_carbon_correction(logg, feh, cfe, e_logg=0.1, e_feh=0.1,
    e_cfe=0.1, n_samples=100, timeout=60):
    """
    Query the Placco et al. (2014) calculator for one star.

    :returns:
        A dictionary with `correction` (to be added to the observed [C/Fe]),
        `mad` (Monte Carlo median absolute deviation, 0 if the calculator gives
        none, e.g. for a zero correction), `nfe` (the [N/Fe] case), `clipped`
        (list of inputs outside the calculator's grid) and `message`.
    """

    values = {"logg": float(logg), "[Fe/H]": float(feh), "[C/Fe]": float(cfe)}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("log(g), [Fe/H] and [C/Fe] must all be finite")

    def _err(value):
        value = float(value) if value is not None else np.nan
        return value if np.isfinite(value) and value > 0 else 0.1

    data = {
        "lgg": "{:.3f}".format(values["logg"]), "e_lgg": "{:.3f}".format(_err(e_logg)),
        "feh": "{:.3f}".format(values["[Fe/H]"]), "e_feh": "{:.3f}".format(_err(e_feh)),
        "cfe": "{:.3f}".format(values["[C/Fe]"]), "e_cfe": "{:.3f}".format(_err(e_cfe)),
        "n_samples": str(int(min(max(n_samples, 1), 1000))),
    }
    response = requests.post(PLACCO_URL, data=data, timeout=timeout)
    response.raise_for_status()

    text = re.sub(r"<[^>]+>", " ", response.text)
    text = re.sub(r"\s+", " ", text)
    match = _RESULT.search(text)
    if match is None:
        raise ValueError("Could not parse the carbon correction from "
                         "vplacco.pythonanywhere.com: {}".format(text[:200]))

    clipped = []
    for name, (low, high) in PLACCO_LIMITS.items():
        if values[name] < low or values[name] > high:
            clipped.append("{} = {:.2f} outside {:g} to {:g} (clipped to grid edge)".format(
                name, values[name], low, high))

    return {
        "correction": float(match.group(1)),
        "mad": float(match.group(2)) if match.group(2) is not None else 0.0,
        "nfe": float(match.group(3)) if match.group(3) is not None else np.nan,
        "clipped": clipped,
        "inputs": dict(values, e_logg=_err(e_logg), e_feh=_err(e_feh),
                       e_cfe=_err(e_cfe), n_samples=int(data["n_samples"])),
        "message": "; ".join(clipped),
        "reference": PLACCO_REFERENCE,
    }
