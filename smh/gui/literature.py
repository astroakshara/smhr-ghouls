#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Literature comparison tab for SMHR abundances."""  #Akshara edits

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import csv
import logging
import os
import re

import numpy as np
from PySide2 import (QtCore, QtWidgets as QtGui)
from matplotlib.ticker import MaxNLocator

import smh
from smh.gui import mpl
from smh import utils

logger = logging.getLogger(__name__)
logger.addHandler(smh.handler)

SAGA_FILENAME = "txt_table_recommended_v2021_MW.tsv"


def _to_float(value):
    try:
        text = str(value).strip()
    except Exception:
        return np.nan
    if text == "" or text.lower() in ("nan", "none", "null"):
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _is_limit(value):
    text = str(value).strip()
    return text in ("<", ">")


def _species_label(species):
    try:
        return utils.species_to_element(float(species))
    except Exception:
        return str(species)


def _element_base(label):
    return str(label).split()[0]


class LiteratureTab(QtGui.QWidget):
    def __init__(self, parent):
        super(LiteratureTab, self).__init__(parent)
        self.parent = parent
        self._saga_rows = None
        self._saga_columns = set()

        layout = QtGui.QVBoxLayout(self)
        controls = QtGui.QHBoxLayout()
        self.label_status = QtGui.QLabel(self)
        self.label_status.setText("Literature comparison")
        self.btn_refresh = QtGui.QPushButton("Refresh", self)
        self.btn_refresh.clicked.connect(self.refresh_plot)
        controls.addWidget(self.label_status)
        controls.addItem(QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding,
                                           QtGui.QSizePolicy.Minimum))
        controls.addWidget(self.btn_refresh)
        layout.addLayout(controls)

        self.scroll = QtGui.QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.figure = mpl.MPLWidget(None)
        self.figure.setMinimumHeight(900)
        self.figure.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                  QtGui.QSizePolicy.Expanding)
        self.scroll.setWidget(self.figure)
        layout.addWidget(self.scroll)

    def new_session_loaded(self):
        self.refresh_plot()
        return None

    def refresh_plot(self):
        self.figure.figure.clear()
        session = self.parent.session
        if session is None:
            self.label_status.setText("Load a session to compare literature abundances")
            self.figure.draw()
            return None

        try:
            rows = self._load_saga_rows()
        except Exception as e:
            self.label_status.setText("Could not load {}: {}".format(SAGA_FILENAME, e))
            logger.exception("Could not load literature table")
            self.figure.draw()
            return None

        points = self._current_star_points(session)
        if len(points) == 0:
            self.label_status.setText("No measured abundances available for literature comparison")
            self.figure.draw()
            return None

        num_plots = len(points)
        num_cols = min(num_plots, 3)
        num_rows = int(np.ceil(num_plots / float(num_cols)))
        self.figure.figure.set_size_inches(5.2*num_cols, 3.8*num_rows + 0.8)
        axes = []

        legend_handles = {}
        for i, point in enumerate(points):
            ax = self.figure.figure.add_subplot(num_rows, num_cols, i + 1)
            axes.append(ax)
            lit = self._literature_values(rows, point["label"])
            if lit is not None and len(lit["feh"]) > 0:
                sc = ax.scatter(lit["feh"], lit["xfe"], s=10, marker="o",
                                color="#222222", alpha=0.35, linewidths=0,
                                label="MW SAGA")
                legend_handles.setdefault("MW SAGA", sc)

            if np.isfinite(point["xfe_lte"]):
                lte = ax.errorbar(point["feh_lte"], point["xfe_lte"],
                                  xerr=point["feh_lte_err"], yerr=point["xfe_lte_err"],
                                  marker="D", markersize=7, linestyle="None",
                                  color="#f4a3a3", ecolor="#f4a3a3",
                                  markeredgecolor="k", markeredgewidth=0.5,
                                  capsize=2, label="Current LTE", zorder=20)
                legend_handles.setdefault("Current LTE", lte)
            if np.isfinite(point["xfe_nlte"]):
                nlte = ax.errorbar(point["feh_nlte"], point["xfe_nlte"],
                                   xerr=point["feh_nlte_err"], yerr=point["xfe_nlte_err"],
                                   marker="D", markersize=7, linestyle="None",
                                   color="#b30000", ecolor="#b30000",
                                   markeredgecolor="k", markeredgewidth=0.5,
                                   capsize=2, label="Current NLTE", zorder=21)
                legend_handles.setdefault("Current NLTE", nlte)

            ax.text(0.04, 0.08, "[{}/Fe]".format(point["label"]),
                    transform=ax.transAxes, fontsize=10)
            ax.set_xlabel("[Fe/H]")
            ax.set_ylabel("[{}/Fe]".format(point["label"]))
            ax.minorticks_on()
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="both", which="major", length=5, width=0.8,
                           direction="in", top=True, right=True)
            ax.tick_params(axis="both", which="minor", length=3, width=0.6,
                           direction="in", top=True, right=True)

        if len(legend_handles) > 0:
            self.figure.figure.legend(list(legend_handles.values()),
                                      list(legend_handles.keys()),
                                      loc="upper center", bbox_to_anchor=(0.5, 0.985),
                                      ncol=min(3, len(legend_handles)),
                                      frameon=False, borderaxespad=0.0)
        self.figure.figure.tight_layout(rect=(0, 0, 1, 0.91))
        self.label_status.setText("{} panels from {}; {} literature stars loaded".format(
            num_plots, os.path.basename(self._saga_path()), len(rows)))
        self.figure.draw()
        return None

    def _saga_path(self):
        here = os.path.abspath(os.path.dirname(__file__))
        candidates = [
            os.path.abspath(os.path.join(here, "..", "..", SAGA_FILENAME)),
            os.path.abspath(os.path.join(here, "..", "..", "..", SAGA_FILENAME)),
            os.path.abspath(os.path.join(os.getcwd(), SAGA_FILENAME)),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _load_saga_rows(self):
        if self._saga_rows is not None:
            return self._saga_rows
        path = self._saga_path()
        rows = []
        with open(path, "r") as fp:
            reader = csv.DictReader(fp, delimiter="\t")
            self._saga_columns = set(reader.fieldnames or [])
            for row in reader:
                rows.append(row)
        self._saga_rows = rows
        return rows

    def _literature_columns_for_label(self, label):
        candidates = [label, _element_base(label)]
        for candidate in candidates:
            value_col = "[{}/H]".format(candidate)
            flag_col = "f({})".format(candidate)
            err_col = "d({})".format(candidate)
            if value_col in self._saga_columns:
                return value_col, flag_col, err_col
        return None

    def _literature_values(self, rows, label):
        cols = self._literature_columns_for_label(label)
        if cols is None or "[Fe I/H]" not in self._saga_columns:
            return None
        value_col, flag_col, err_col = cols
        feh_values, xfe_values = [], []
        for row in rows:
            if _is_limit(row.get("f(Fe I)", "")) or _is_limit(row.get(flag_col, "")):
                continue
            feh = _to_float(row.get("[Fe I/H]", ""))
            xh = _to_float(row.get(value_col, ""))
            err = _to_float(row.get(err_col, ""))
            if not np.isfinite(feh*xh):
                continue
            if not (-3.15 < feh < -1.5):
                continue
            if np.isfinite(err) and err >= 0.2:
                continue
            feh_values.append(feh)
            xfe_values.append(xh - feh)
        return {"feh": np.array(feh_values), "xfe": np.array(xfe_values)}

    def _current_star_points(self, session):
        summary = session.summarize_spectral_models(what_fe=1)
        nlte_summary = session.summarize_spectral_models(what_fe=1, use_nlte=True)
        fe_lte = self._fe_summary(summary)
        fe_nlte = self._fe_summary(nlte_summary)
        if fe_lte is None and fe_nlte is None:
            return []
        feh_lte, feh_lte_err = fe_lte if fe_lte is not None else (np.nan, np.nan)
        feh_nlte, feh_nlte_err = fe_nlte if fe_nlte is not None else (feh_lte, feh_lte_err)

        points = []
        for species in sorted(summary.keys()):
            try:
                if int(float(species)) == 26:
                    continue
            except Exception:
                pass
            label = _species_label(species)
            if self._literature_columns_for_label(label) is None:
                continue
            vals = summary.get(species, [np.nan]*6)
            nlte_vals = nlte_summary.get(species, [np.nan]*6)
            points.append({
                "species": species,
                "label": label,
                "feh_lte": feh_lte,
                "feh_lte_err": feh_lte_err,
                "feh_nlte": feh_nlte,
                "feh_nlte_err": feh_nlte_err,
                "xfe_lte": vals[5],
                "xfe_lte_err": vals[3],
                "xfe_nlte": nlte_vals[5] if len(nlte_vals) > 5 else np.nan,
                "xfe_nlte_err": nlte_vals[3] if len(nlte_vals) > 3 else np.nan,
            })
        return points

    def _fe_summary(self, summary):
        for species in (26.0, 26.1):
            if species in summary:
                vals = summary[species]
                return vals[4], vals[3]
        return None
