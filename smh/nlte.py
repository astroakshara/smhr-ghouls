#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Approximate corrections to non-local thermodynamic equilibrium conditions. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import os
import re
import requests
import io  #Akshara edits
import zipfile  #Akshara edits

from astropy.table import Table
from glob import glob
from scipy import interpolate


MPIA_URL = "https://nlte.mpia.de/gui-siuAC_secE.php"  #Akshara edits
MPIA_SPECIAL_URL = "https://nlte.mpia.de/gui-siuAC_secEnew.php"
MPIA_ELEMENTS = (8.01, 12.01, 14.01, 20.01, 20.02, 22.01,
    22.02, 24.01, 25.01, 26.01, 26.02, 27.01)
MPIA_ERROR_CODES = {
    30.0: "Line not in linelist or too weak",
    20.0: "NLTE not converged",
    10.0: "Error in line formation",
    0.0: "No NLTE departures for this line"
}
INSPECT_URL = "http://www.inspect-stars.com/"
INSPECT_ELEMENTS = ("Li", "O", "Na", "Mg", "Ti", "Fe", "Sr")
MPIA_MAX_LINES_PER_REQUEST = 99
NLITE_URL = "https://nlite.pythonanywhere.com/"  #Akshara edits
# Preferred NLiTE grids per species (grid numbers as listed on the NLiTE
# website), tried in order: if the star is outside a grid, the next one is
# used. Choices follow collaborator recommendations; species not listed use
# the NLiTE reference grid (1).  #Akshara edits
NLITE_PREFERRED_GRIDS = {
    "LiI": [1],      # Wang et al. 2021 (3D NLTE)
    "CI": [1],       # Amarsi et al. 2019 (1D NLTE)
    "CH": [1],       # Norris et al. 2019 (1D; only CH grid in NLiTE)
    "OI": [1],       # Amarsi et al. 2019 (1D NLTE)
    "NaI": [1],      # Lind et al. 2022 (all lines)
    "MgI": [4],      # Lind et al. 2022
    "AlI": [1],      # Lind et al. 2022, resonance lines (4 = subordinate lines)
    "SiI": [1],      # Amarsi et al. 2017
    "KI": [1],       # Reggiani et al. 2019
    "TiI": [1],      # Bergemann et al. 2011 (Mallinson et al. not in NLiTE)
    "TiII": [1],     # Bergemann et al. 2011 (Mallinson et al. not in NLiTE)
    "FeI": [4, 1],   # Amarsi et al. 2022 (3D, dwarfs), else mean Mashonkina+11/Bergemann+12 (1D)
    "FeII": [2, 1],  # Amarsi et al. 2022 (3D, dwarfs), else Bergemann et al. 2012 (1D)
    "CuI": [1],      # Andrievsky+18/Shi+18/Xu+22 fits (Caliskan et al. not in NLiTE)
    "ZnI": [1],      # Sitnova et al. 2022
    "SrII": [1],     # Mashonkina et al. 2022
    "BaII": [1],     # Mashonkina et al. 2019
}
NLITE_AL_SUBORDINATE_GRID = 4  #Akshara edits: Lind et al. 2022, 6696-8773 A lines


def parse_stellar_parameters(path):
    """
    Parse the element and stellar parameters from a pre-computed set of non-LTE
    corrections.

    :param path:
        The filename path (e.g., "EW_out.fe.nlte.marcs_4000_+1.0_-2.00_00.MULTI")

    :returns:
        A four-length tuple containing the temperature, surface gravity,
        metallicity, and element.
    """

    basename = os.path.basename(path)
    element = basename.split(".")[1].title()

    _, __, teff, logg, feh, ___ = basename.split("_")
    return (float(teff), float(logg), float(feh), element)


def parse_transitions(path, fast=False, **kwargs):
    """
    Parse transitions from a pre-computed file of non-LTE corrections.

    :param fast: [optional]
        If `True`, `numpy.loadtxt` is used to load the data rather than 
        returning back an `astropy.Table`.
    """

    # Load in the transitions.
    if fast:
        return np.loadtxt(path, **kwargs)

    return Table.read(path, names=("wavelength", "expot", "g_lower_level", 
        "EW_nlte", "EW_lte", "EW_delta", "departure_coefficient"), format="ascii")


def mpia_model_for_smhr(model_atmosphere=None):  #Akshara edits
    """
    Return an MPIA model-atmosphere identifier.

    SMHR defaults to Castelli/Kurucz photospheres, which MPIA does not expose.
    For that default case we use MPIA's default MAFAGS-OS grid. If callers pass
    a supported MPIA model name, it is preserved.
    """

    if model_atmosphere is None:
        return "mafags-os"

    model_atmosphere = str(model_atmosphere).lower()
    if model_atmosphere in ("mafags-os", "marcs", "rsg", "3dmean"):
        return model_atmosphere
    if model_atmosphere == "castelli/kurucz":
        return "mafags-os"
    return "mafags-os"


def _strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def _parse_mpia_response(text):
    text = _strip_html(text)
    if "lines(A)" not in text or "my_star" not in text:
        raise ValueError("MPIA response did not contain NLTE correction output")

    result = text.split("lines(A)", 1)[-1].split("Download", 1)[0]
    line_text, delta_text = result.split("my_star", 1)
    lines = [float(value) for value in line_text.split()]
    deltas = [float(value) for value in delta_text.split()[:len(lines)]]
    if len(lines) != len(deltas):
        raise ValueError("MPIA response line/delta lengths did not match")

    cleaned_deltas = []
    messages = []
    for line, delta in zip(lines, deltas):
        message = MPIA_ERROR_CODES.get(delta)
        if message is None:
            cleaned_deltas.append(delta)
            messages.append("")
        else:
            cleaned_deltas.append(np.nan)
            messages.append("{} at {:.3f}".format(message, line))
    return np.array(lines), np.array(cleaned_deltas), messages


def mpia_nlte_corrections(element, lines, effective_temperature,
    surface_gravity, metallicity, microturbulence, model_atmosphere=None,
    timeout=60):
    """
    Request NLTE abundance corrections from the MPIA online service.

    Returns a dictionary with `element`, `line`, `delta`, `message`, and
    `model_atmosphere`. Invalid service/error-code deltas are returned as NaN.
    """

    lines = np.array(lines, dtype=float)
    if len(lines) == 0:
        return {
            "element": float(element),
            "line": lines,
            "delta": np.array([], dtype=float),
            "message": [],
            "model_atmosphere": mpia_model_for_smhr(model_atmosphere)
        }

    element = float(element)
    if element not in MPIA_ELEMENTS:
        return {
            "element": element,
            "line": lines,
            "delta": np.nan * np.ones(len(lines)),
            "message": ["Element not available from MPIA"] * len(lines),
            "model_atmosphere": mpia_model_for_smhr(model_atmosphere)
        }

    model_atmosphere = mpia_model_for_smhr(model_atmosphere)
    url = MPIA_URL
    if element in (8.01, 25.01, 27.01):
        url = MPIA_SPECIAL_URL
        model_atmosphere = "mafags-os"

    if model_atmosphere == "marcs":
        limits = ((2500, 7750), (-0.5, 3.5), (-5.0, 1.0))
        params = (effective_temperature, surface_gravity, metallicity)
        if any(param < lo or param > hi for param, (lo, hi) in zip(params, limits)):
            model_atmosphere = "mafags-os"
    elif model_atmosphere == "rsg":
        limits = ((3400, 4400), (-0.99, 1.0), (-1.5, 1.0))
        params = (effective_temperature, surface_gravity, metallicity)
        if any(param < lo or param > hi for param, (lo, hi) in zip(params, limits)):
            model_atmosphere = "mafags-os"

    data = {
        "model": model_atmosphere,
        "user_input": "my_star {} {} {} {}".format(
            effective_temperature, surface_gravity, metallicity,
            microturbulence),
        "lines_input": "\n".join(
            "{} {}".format(line, element) for line in lines)
    }
    response = requests.post(url, data=data, timeout=timeout)
    response.raise_for_status()

    response_lines, deltas, messages = _parse_mpia_response(response.text)
    return {
        "element": element,
        "line": response_lines,
        "delta": deltas,
        "message": messages,
        "model_atmosphere": model_atmosphere
    }



def request_mpia_nlte_corrections(effective_temperature, surface_gravity,
    metallicity, microturbulence, alpha, species, abundances, equivalent_widths,
    loggf, wavelengths, excitation_potentials, model_atmosphere=None,
    timeout=60, max_lines_per_request=MPIA_MAX_LINES_PER_REQUEST):
    """Request MPIA corrections in <=99-line batches.

    The extra abundance/EW/loggf/expot arguments are accepted for API symmetry
    with other providers; MPIA only uses species and wavelengths.
    """

    wavelengths = np.array(wavelengths, dtype=float)
    abundances = np.array([float(np.ravel(abundance)[0])
                           for abundance in abundances], dtype=float)
    n_lines = len(wavelengths)
    deltas = np.nan * np.ones(n_lines)
    messages = [""] * n_lines
    response_lines = np.nan * np.ones(n_lines)
    model_name = mpia_model_for_smhr(model_atmosphere)

    if n_lines == 0:
        return {
            "delta": deltas,
            "abundance_nlte": deltas,
            "message": messages,
            "line": response_lines,
            "model_atmosphere": model_name
        }

    for start in range(0, n_lines, max_lines_per_request):
        stop = min(start + max_lines_per_request, n_lines)
        batch = slice(start, stop)
        try:
            result = mpia_nlte_corrections(
                species, wavelengths[batch], effective_temperature,
                surface_gravity, metallicity, microturbulence,
                model_atmosphere=model_atmosphere, timeout=timeout)
        except Exception as e:
            messages[start:stop] = ["MPIA request failed: {}".format(e)] * (stop - start)
            continue

        batch_deltas = np.array(result.get("delta", []), dtype=float)
        n_returned = min(len(batch_deltas), stop - start)
        deltas[start:start+n_returned] = batch_deltas[:n_returned]
        response_line_values = np.array(result.get("line", []), dtype=float)
        if len(response_line_values) >= n_returned:
            response_lines[start:start+n_returned] = response_line_values[:n_returned]
        else:
            response_lines[start:stop] = wavelengths[batch]
        batch_messages = list(result.get("message", []))
        for i in range(n_returned):
            messages[start+i] = batch_messages[i] if i < len(batch_messages) else ""
        if n_returned < stop - start:
            for i in range(start+n_returned, stop):
                messages[i] = "MPIA returned fewer corrections than requested"
        model_name = result.get("model_atmosphere", model_name)

    abundance_nlte = abundances + deltas
    abundance_nlte[~np.isfinite(deltas)] = np.nan
    return {
        "delta": deltas,
        "abundance_nlte": abundance_nlte,
        "message": messages,
        "line": response_lines,
        "model_atmosphere": model_name
    }


def request_inspect_nlte_correction(element, effective_temperature,
    surface_gravity, metallicity, microturbulence, abundance, wavelength,
    expot=None, ew=None, line=None, timeout=60):
    """Compatibility wrapper around INSPECT's one-line query."""

    abundance_lte = float(np.ravel(abundance)[0])
    return inspect_nlte_correction(
        element, wavelength, abundance_lte, effective_temperature,
        surface_gravity, metallicity, microturbulence, timeout=timeout)

def _inspect_line_index(line, line_options, tolerance=0.2):
    if len(line_options) == 0:
        return None
    wavelengths = np.array([float(key) for key in line_options.keys()])
    index = np.argmin(np.abs(wavelengths - float(line)))
    if np.abs(wavelengths[index] - float(line)) <= tolerance:
        return line_options[list(line_options.keys())[index]]
    return None


def _inspect_bounds(html):
    bounds = []
    pattern = r'class=""></td><td>.*?</td></tr>'
    for match in re.finditer(pattern, html):
        text = html[match.start():match.end()]
        text = text.strip('class=""></td></tr>[]').split(',')
        try:
            bounds.append([float(value) for value in text])
        except ValueError:
            continue
    return bounds


def inspect_nlte_correction(element, line, abundance_lte,
    effective_temperature, surface_gravity, metallicity, microturbulence,
    timeout=60):
    """
    Request an INSPECT NLTE abundance correction for one LTE abundance.

    Returns a dictionary with `delta`, `abundance_nlte`, and `message`. Failed
    or unavailable corrections are returned as NaN.
    """

    element = str(element)
    if element not in INSPECT_ELEMENTS:
        return {
            "element": element,
            "line": float(line),
            "delta": np.nan,
            "abundance_nlte": np.nan,
            "message": "Element not available from INSPECT"
        }

    elem_url = INSPECT_URL + "nonlte_from_lte?element_name=" + element
    response = requests.get(elem_url, timeout=timeout)
    response.raise_for_status()
    html = response.text

    line_options = {}
    for match in re.finditer(r"<option.*?>.*?</option.*?>", html):
        option = html[match.start():match.end()]
        option = option.strip('<option value"></=').replace('"', '').split(">")
        if len(option) > 1:
            try:
                line_options[option[1]] = int(option[0])
            except ValueError:
                continue

    line_index = _inspect_line_index(line, line_options)
    if line_index is None:
        return {
            "element": element,
            "line": float(line),
            "delta": np.nan,
            "abundance_nlte": np.nan,
            "message": "Line not available from INSPECT"
        }

    inputs = [abundance_lte, effective_temperature, surface_gravity,
              metallicity, microturbulence]
    names = ["A(X)", "Teff", "logg", "[Fe/H]", "vt"]
    if element in ("Ti", "Fe"):
        inputs = [abundance_lte, effective_temperature, surface_gravity,
                  microturbulence]
        names = ["A(X)", "Teff", "logg", "vt"]
    bounds = _inspect_bounds(html)
    for name, value, bound in zip(names, inputs, bounds):
        if value < bound[0] or value > bound[1]:
            return {
                "element": element,
                "line": float(line),
                "delta": np.nan,
                "abundance_nlte": np.nan,
                "message": "{} outside INSPECT bounds [{}, {}]".format(
                    name, bound[0], bound[1])
            }

    params = {
        "A_lte": abundance_lte,
        "t": effective_temperature,
        "g": surface_gravity,
        "f": metallicity,
        "x": microturbulence,
        "wi": line_index
    }
    response = requests.get(elem_url, params=params, timeout=timeout)
    response.raise_for_status()
    html = response.text
    if "Calculation failed" in html:
        return {
            "element": element,
            "line": float(line),
            "delta": np.nan,
            "abundance_nlte": np.nan,
            "message": "INSPECT calculation failed"
        }

    try:
        row = html.split("pre")[1].split("\n")[3].split("\t")
        values = [float(value) for value in row]
        abundance_nlte = values[2]
        delta = values[3]
    except Exception:
        return {
            "element": element,
            "line": float(line),
            "delta": np.nan,
            "abundance_nlte": np.nan,
            "message": "Could not parse INSPECT response"
        }

    if not np.isfinite(delta):
        abundance_nlte = np.nan
    return {
        "element": element,
        "line": float(line),
        "delta": delta,
        "abundance_nlte": abundance_nlte,
        "message": ""
    }


def nlite_label(species):  #Akshara edits
    """
    Convert an SMHR species (e.g. 26.1) into an NLiTE column label
    (e.g. "FeII"). NLiTE accepts element names with or without the ion,
    e.g. "Fe", "FeI", "FeII".
    """
    from .utils import species_to_element
    if round(float(species), 1) == 106.0:  # CH molecule  #Akshara edits
        return "CH"
    return species_to_element(round(float(species), 1)).replace(" ", "")


def _nlite_form_errors(html):  #Akshara edits
    errors = re.findall(r'<span style="color: red;">(.*?)</span>', html, re.S)
    errors = [_strip_html(error).strip() for error in errors]
    return [error for error in errors if error]


def _nlite_read_zip(content):  #Akshara edits
    """ Return (corrections table, references text) from an NLiTE zip. """
    archive = zipfile.ZipFile(io.BytesIO(content))
    table, references = None, ""
    for name in archive.namelist():
        if name.endswith("_NLTE_corrections.tsv"):
            text = archive.read(name).decode("utf-8")
            table = Table.read(text, format="ascii.tab", fill_values=[("", "nan")])
        elif name.endswith("_References.txt"):
            references = archive.read(name).decode("utf-8", "replace")
    if table is None:
        raise ValueError("NLiTE zip did not contain an NLTE corrections table")
    return table, references


def _nlite_request(rows, labels, grid=None, timeout=120):  #Akshara edits
    """
    Send rows of (Teff, logg, [Fe/H], [label/H]...) to NLiTE and return the
    (corrections table, references text). With grid=None the "all elements"
    reference-grid mode is used; otherwise the one-element grid-of-choice
    mode (one label only).
    """
    columns = ["Teff", "logg", "[Fe/H]"] + ["[{}/H]".format(label) for label in labels]
    tsv = "\t".join(columns) + "\n" + "".join(
        "\t".join("{:.4f}".format(float(value)) for value in row) + "\n"
        for row in rows)

    if grid is None:
        files = {"another_file": ("smhr_nlite_input.tsv", tsv, "text/tab-separated-values")}
        data = {"action_all": "Compute Corrections"}
    else:
        files = {"input_file": ("smhr_nlite_input.tsv", tsv, "text/tab-separated-values")}
        data = {"action": "Compute Corrections", "el": labels[0],
                "option": str(int(grid))}

    response = requests.post(NLITE_URL, data=data, files=files,
                             allow_redirects=False, timeout=timeout)
    if response.status_code not in (301, 302, 303):
        response.raise_for_status()
        errors = _nlite_form_errors(response.text)
        raise ValueError("NLiTE did not compute corrections{}".format(
            ": " + "; ".join(errors) if errors else ""))

    # The redirect target carries the job id, element and grid option needed
    # to download the results.
    location = response.headers.get("Location", "")
    query = dict(re.findall(r"[?&]([^=&]+)=([^&]*)", location))
    if "base_name" not in query:
        raise ValueError("Could not parse NLiTE redirect: {}".format(location))
    params = {"base_name": requests.utils.unquote(query["base_name"]),
              "el": requests.utils.unquote(query.get("element", "all elements").replace("+", " ")),
              "option": requests.utils.unquote(query.get("option", "-99"))}
    response = requests.get(NLITE_URL + "download_zip", params=params,
                            timeout=timeout)
    response.raise_for_status()
    return _nlite_read_zip(response.content)


def nlite_nlte_corrections(effective_temperature, surface_gravity, metallicity,
    abundances_xh, grid=None, timeout=120):  #Akshara edits
    """
    Request mean (per element/ion) NLTE corrections from NLiTE
    (https://nlite.pythonanywhere.com/; Koutsouridou et al. 2025), which
    interpolates within precomputed grids of NLTE corrections averaged over
    the commonly used lines of each species. Intended for metal-poor stars
    ([Fe/H] <= -1).

    :param abundances_xh:
        A dictionary of {NLiTE label: [X/H]_LTE}, e.g. {"BaII": -3.0}.

    :param grid: [optional]
        NLTE grid number (as listed on the NLiTE website). If None, the
        "correct all elements" mode with the reference (1) grids is used and
        all labels are sent in one request. If given, the "one element with
        grid of choice" mode is used, so only one label may be supplied.

    :returns:
        A dictionary {"corrections": {label: {"delta", "in_grid", "std",
        "nlines"}}, "references": str}. Labels without an NLiTE grid are
        omitted from the corrections.
    """

    labels = [label for label, xh in abundances_xh.items()
              if xh is not None and np.isfinite(xh)]
    if len(labels) == 0:
        return {"corrections": {}, "references": ""}
    if grid is not None and len(labels) != 1:
        raise ValueError("NLiTE grid-of-choice mode takes exactly one element")

    row = [effective_temperature, surface_gravity, metallicity] \
        + [abundances_xh[label] for label in labels]
    table, references = _nlite_request(
        [row], labels, grid=grid, timeout=timeout)  #Akshara edits: shared request helper

    def _value(column):
        if column not in table.colnames:
            return np.nan
        try:
            value = table[column][0]
            if np.ma.is_masked(value):  #Akshara edits
                return np.nan
            return float(value)
        except Exception:
            return np.nan

    corrections = {}
    for label in labels:
        key = "[{}/H]".format(label)
        delta = _value("d{}_NLTE".format(key))
        if not np.isfinite(delta):
            continue
        corrections[label] = {
            "delta": delta,
            "in_grid": bool(_value("in_grid_{}".format(key)) == 1),
            "std": _value("std_d{}_NLTE".format(key)),
            "nlines": _value("Nlines_d{}_NLTE".format(key)),
        }
    return {"corrections": corrections, "references": references}


def nlite_diagnose_out_of_grid(effective_temperature, surface_gravity,
    metallicity, label, abundance_xh, grid=1, timeout=180):  #Akshara edits
    """
    Work out which stellar parameter(s) put a star outside an NLiTE grid.

    NLiTE only reports an in_grid flag, so each parameter (Teff, logg,
    [Fe/H], [X/Fe]) is scanned on its own, holding the others at the star's
    values, in a single multi-row NLiTE request. [Fe/H] is scanned at fixed
    [X/Fe] and [X/Fe] at fixed [Fe/H]. A parameter whose scan reaches the
    grid is one that pushes the star out of it, and the scan gives the
    approximate in-grid range for that parameter. If no single parameter
    works, a coarse scan over every pair of parameters finds the nearest
    in-grid point.

    :returns:
        A dictionary with `culprits` (list of parameter names), `ranges`
        ({name: (low, high)}), `nearest` (for the two-parameter case,
        {name: value}), `values` (the star's values) and `message`.
    """

    teff, logg, feh = float(effective_temperature), float(surface_gravity), \
        float(metallicity)
    xfe = float(abundance_xh) - feh
    star = [teff, logg, feh, xfe]
    names = ["Teff", "logg", "[Fe/H]", "[{}/Fe]".format(label)]
    formats = ["{:.0f}", "{:.2f}", "{:.2f}", "{:+.2f}"]
    scales = [500., 1.0, 0.5, 0.5]  # for "nearest" in the two-parameter scan

    def to_row(params):  # (Teff, logg, [Fe/H], [X/Fe]) -> NLiTE row
        return [params[0], params[1], params[2], params[2] + params[3]]

    def query(points):
        table, _ = _nlite_request([to_row(p) for p in points], [label],
                                  grid=grid, timeout=timeout)
        column = "in_grid_[{}/H]".format(label)
        if column not in table.colnames or len(table) != len(points):
            raise ValueError("Unexpected NLiTE response while diagnosing grid limits")
        return np.array([(not np.ma.is_masked(v)) and np.isfinite(float(v))
                         and float(v) == 1 for v in table[column]], dtype=bool)

    fine = [np.arange(3500., 8000.1, 100.),
            np.round(np.arange(-0.5, 5.51, 0.1), 2),
            np.round(np.arange(-5.0, 0.51, 0.1), 2),
            np.round(np.arange(xfe - 3.0, xfe + 3.01, 0.1), 2)]
    steps = [100., 0.1, 0.1, 0.1]
    points, owners = [list(star)], [-1]
    for i, values in enumerate(fine):
        for value in values:
            point = list(star)
            point[i] = value
            points.append(point)
            owners.append(i)
    in_grid = query(points)
    owners = np.array(owners)

    culprits, ranges, reasons = [], {}, []
    for i, name in enumerate(names):
        values = np.array([p[i] for p in points])[owners == i]
        flags = in_grid[owners == i]
        if not np.any(flags):
            continue
        runs, current = [], []
        for value, flag in zip(values, flags):
            if flag:
                current.append(value)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        run = min(runs, key=lambda run: min(abs(v - star[i]) for v in run))
        low, high = min(run), max(run)
        ranges[name] = (low, high)
        culprits.append(name)
        direction = "too low" if star[i] < low else "too high" if star[i] > high \
            else "between in-grid regions"
        fmt = formats[i]
        reasons.append("{} = {} {} (in grid for ~{} to {})".format(
            name, fmt.format(star[i]), direction, fmt.format(low), fmt.format(high)))

    nearest = {}
    if len(culprits):
        message = "; ".join(reasons)
        if len(culprits) > 1:
            message += " [changing any one of these alone is enough]"
        message += " (other parameters fixed)"
    else:
        # Coarse scan over every pair of parameters
        coarse = [np.arange(3500., 8000.1, 250.),
                  np.round(np.arange(-0.5, 5.51, 0.5), 2),
                  np.round(np.arange(-5.0, 0.51, 0.25), 2),
                  np.round(np.arange(xfe - 2.0, xfe + 2.01, 0.25), 2)]
        pair_points, pair_owner = [], []
        for i in range(4):
            for j in range(i + 1, 4):
                for a in coarse[i]:
                    for b in coarse[j]:
                        point = list(star)
                        point[i], point[j] = a, b
                        pair_points.append(point)
                        pair_owner.append((i, j))
        pair_in_grid = query(pair_points)
        best, best_distance = None, np.inf
        for point, (i, j), flag in zip(pair_points, pair_owner, pair_in_grid):
            if not flag:
                continue
            distance = np.hypot((point[i] - star[i]) / scales[i],
                                (point[j] - star[j]) / scales[j])
            if distance < best_distance:
                best, best_distance = (point, i, j), distance
        if best is None:
            message = ("Outside the grid in 3 or more parameters together "
                       "(no in-grid point found changing one or two of Teff, "
                       "logg, [Fe/H], [X/Fe])")
        else:
            point, i, j = best
            culprits = [names[i], names[j]]
            nearest = {names[i]: point[i], names[j]: point[j]}
            message = ("{} and {} together: nearest in-grid point is {} = {} "
                       "(star {}), {} = {} (star {}) (coarse scan)").format(
                names[i], names[j],
                names[i], formats[i].format(point[i]), formats[i].format(star[i]),
                names[j], formats[j].format(point[j]), formats[j].format(star[j]))
    return {"culprits": culprits, "ranges": ranges, "nearest": nearest,
            "values": dict(zip(names, star)), "message": message}

class ApproximateNLTECorrectionsBase(object):
    pass


class ApproximateNLTECorrections(ApproximateNLTECorrectionsBase):

    _minimum_paths = 1

    def __init__(self, precomputed_corrections_wildmask):
        """
        Initialize an object to approximate (interpolate) between grids of pre-
        computed non-LTE corrections.

        :param precomputed_corrections_wildmask:
            A wildmask for paths matching files that contain non_LTE corrections
            to include in this class.
        """

        self._wildmask = precomputed_corrections_wildmask
        self._paths = glob(self._wildmask)

        if len(self._paths) < self._minimum_paths:
            raise ValueError(
                "could not find enough computed corrections: ({} < {})".format(
                    len(self._paths), self._minimum_paths))

        # Load the gridpoints
        self._grid = []
        self._element = []

        for path in self._paths:
            teff, logg, mh, element = parse_stellar_parameters(path)
            self._grid.append([teff, logg, mh])
            self._element.append(element)

        if len(set(self._element)) > 1:
            raise ValueError(
                "separate ApproximateNLTECorrections classes must be set up for"
                " each element")

        self._element, self._grid = self._element[0], np.array(self._grid)

        # Load the transitions from one file.
        self._transitions = parse_transitions(self._paths[0])

        # Get the EW_delta for each transition.
        self._ew_delta = np.array([
            parse_transitions(path, fast=True, usecols=(5, )) \
            for path in self._paths])

        return None



    def match_transition(self, wavelength, expot, wavelength_tolerance=0.01, 
        expot_tolerance=0.01, **kwargs):

        return \
            (np.abs(self._transitions["wavelength"] - wavelength) < wavelength_tolerance) \
          * (np.abs(self._transitions["expot"] - expot) < expot_tolerance)



    def neighbours(self, effective_temperature, surface_gravity, metallicity, N,
        scales=None):
        """
        Return indices of the `N`th-nearest neighbours in the grid. The three
        parameters are scaled by the peak-to-peak range in the grid, unless
        `scales` are indicates.

        :param effective_temperature:
            The effective temperature of the star.

        :param surface_gravity:
            The surface gravity of the star.

        :param metallicity:
            The metallicity of the star.

        :param N:
            The number of neighbouring indices to return.

        :returns:
            An array of length `N` that contains the indices of the closest
            neighbours in the grid.
        """

        point = np.array([effective_temperature, surface_gravity, metallicity])
        if scales is None:
            scales = np.ptp(self._grid, axis=0)

        distance = np.sum(((self._grid - point)/scales)**2, axis=1)

        return np.argsort(distance)[:N]


    def __call__(self, effective_temperature, surface_gravity, metallicity,
        wavelength, excitation_potential, equivalent_width=None, **kwargs):
        """
        Calculate the interpolated non-LTE correction for a single transition
        given the stellar parameters.

        :param effective_temperature:
            The effective temperature of the star.

        :param surface_gravity:
            The surface gravity of the star.

        :param metallicity:
            The overall metallicity of the star.

        :param wavelength:
            The wavelength of the transition (in Angstroms).

        :param excitation_potential:
            The excitation potential of the transition (in eV).

        :param equivalent_width: [optional]
            The measured equivalent width (in milliAngstroms).

        """

        # Is this transition in our grid?
        matches = self.match_transition(wavelength, excitation_potential,
            **kwargs)

        if not any(matches):
            # No correction available.
            return np.nan

        assert 2 > matches.sum()

        ews = self._ew_delta[:, matches]

        # Find N closest in the grid to interpolate.
        N = kwargs.pop("N", 30)
        neighbouring_indices = self.neighbours(
            effective_temperature, surface_gravity, metallicity, N)

        raise NotImplementedError


    def interpolate(self, stellar_photosphere, transitions):
        """
        Interpolate non-LTE corrections to the equivalent widths of many
        transitions for a single stellar photosphere.

        :param stellar_photosphere:
            A stellar atmosphere model.

        :param transitions:
            A table of atomic transitions.
        """


        # A convenience function.
        raise NotImplementedError



if __name__ == "__main__":

    foo = ApproximateNLTECorrections("EW*")
