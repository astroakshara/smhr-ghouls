#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Approximate corrections to non-local thermodynamic equilibrium conditions. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import os
import re
import requests

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
