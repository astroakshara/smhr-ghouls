#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" The review tab in Spectroscopy Made Hard """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

#__all__ = ["ReviewTab"]

import logging
import numpy as np
import sys
from PySide2 import (QtCore, QtWidgets as QtGui)
import time

import smh
from smh.gui import base
from smh.gui.base import BaseTableView, MeasurementTableView, MeasurementTableDelegate
from smh.gui.base import MeasurementTableModelBase, MeasurementTableModelProxy
from smh.gui.base import MeasurementSummaryTableModel
from smh.gui.base import SMHScatterplot
from smh.gui import mpl, style_utils
from smh import utils

logger = logging.getLogger(__name__)
logger.addHandler(smh.handler)

class ReviewTab(QtGui.QWidget):
    def __init__(self, parent):
        super(ReviewTab, self).__init__(parent)
        self.parent = parent
        self.parent_layout = QtGui.QHBoxLayout(self)
        
        #######################################
        #### LHS: Summary and Measurement Table
        #######################################
        lhs_layout = QtGui.QVBoxLayout()
        self._init_summary_table()
        self._current_species = None
        lhs_layout.addWidget(self.summary_view)
        
        vbox = self._init_measurement_table()
        #lhs_layout.addWidget(self.measurement_view)
        lhs_layout.addLayout(vbox)
        self.parent_layout.addLayout(lhs_layout)
        
        ##################################
        #### RHS: Interactive Scatterplots
        ##################################
        rhs_layout = QtGui.QVBoxLayout()
        rhs_layout.setSpacing(0)
        self._init_scatterplots()
        rhs_layout.addWidget(self.plot1)
        rhs_layout.addWidget(self.plot2)
        rhs_layout.addWidget(self.plot3)
        self.parent_layout.addLayout(rhs_layout)
        return None
    
    def new_session_loaded(self):
        session = self.parent.session
        if session is None: return None
        self.measurement_model.beginResetModel()
        self.summary_model.new_session(session)
        self.full_measurement_model.new_session(session)
        self.measurement_model.reindex()
        self.measurement_model.endResetModel()
        self.measurement_view.update_session(session)
        self.refresh_plots()
        return None
        
    def selected_summary_changed(self):
        """
        Called when you select a new summary
        """
        start = time.time()
        # Get the species
        try:
            row = self.summary_view.selectionModel().selectedRows()[0].row()
        except IndexError:
            species = None
        else:
            species = self.summary_view.model().all_species[row]
        # Change measurement table filter
        logger.debug("species={}".format(species))
        self.change_measurement_table_species(species)
        # Change plots
        self.refresh_plots()
        logger.debug("selected_summary_changed to {}: {:.1f}".format(species, time.time()-start))
        return None
    def change_measurement_table_species(self, species):
        # Update the filter
        self.measurement_model.beginResetModel()
        if self._current_species is not None:
            _current_species_str = "{:.1f}".format(self._current_species)
            try:
                self.measurement_model.delete_filter_function(_current_species_str)
            except KeyError as e:
                logger.debug(e)
                logger.debug(self.measurement_model.filter_functions)
                raise
        if species is not None:
            filterfn = lambda model: _model_has_species(model, species)
            speciesstr = "{:.1f}".format(species)
            self.measurement_model.add_filter_function(speciesstr, filterfn)
        self._current_species = species
        # Reset the model (and its views)
        self.measurement_model.endResetModel()
    def refresh_plots(self):
        sigma = self._get_sigma_to_plot()
        self.plot1.sigma = sigma
        self.plot2.sigma = sigma
        self.plot3.sigma = sigma
        self.plot1.update_scatterplot(True)
        self.plot2.update_scatterplot(True)
        self.plot3.update_scatterplot(True)
        self.plot1.update_selected_points(True)
        self.plot2.update_selected_points(True)
        self.plot3.update_selected_points(True)
        self.update_nlte_overlays()  #Akshara edits
    def refresh_selected_points(self):
        self.plot1.update_selected_points(True)
        self.plot2.update_selected_points(True)
        self.plot3.update_selected_points(True)

    def _get_sigma_to_plot(self):  #Akshara edits
        try:
            return float(self.parent.stellar_parameters_tab.edit_sigma.text())
        except Exception:
            return 2.5

    def update_nlte_overlays(self):  #Akshara edits
        if not hasattr(self, "_nlte_overlay_artists"):
            self._nlte_overlay_artists = []
        for artist in self._nlte_overlay_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._nlte_overlay_artists = []

        if self.parent.session is None:
            return None
        rows = np.arange(self.measurement_model.rowCount())
        models = [model for model in self.measurement_model.get_models_from_rows(rows)
                  if model.is_acceptable and not model.is_upper_limit
                  and np.isfinite(model.abundance_nlte_filled)]
        if len(models) == 0:
            return None

        sigma = self._get_sigma_to_plot()
        for fig, xattr in [(self.plot1, "expot"),
                           (self.plot2, "reduced_equivalent_width"),
                           (self.plot3, "wavelength")]:
            x = np.array([getattr(model, xattr) for model in models], dtype=float)
            y = np.array([model.abundance_nlte_filled for model in models], dtype=float)
            finite = np.isfinite(x*y)
            if not np.any(finite):
                continue

            x, y = x[finite], y[finite]
            color = "#1f77b4"
            artist = fig.ax.scatter(x, y, marker="^", s=24,
                                    facecolor="none", edgecolor=color,
                                    linewidths=1.0, zorder=20)
            self._nlte_overlay_artists.append(artist)

            try:
                m, b, medy, stdy, stdm, N = utils.fit_line(x, y)
            except Exception as e:
                logger.debug("Could not fit NLTE review overlay: {}".format(e))
                continue

            xlim = np.array(fig.ax.get_xlim())
            line = fig.ax.plot(xlim, m*xlim + b, color=color, linestyle="--",
                               linewidth=1.0, zorder=19)[0]
            mean = fig.ax.axhline(medy, color=color, linestyle=":",
                                  linewidth=1.0, zorder=18)
            band = fig.ax.fill_between(xlim, medy - sigma*stdy,
                                       medy + sigma*stdy, color=color,
                                       alpha=0.08, lw=0, zorder=17)
            self._nlte_overlay_artists.extend([line, mean, band])
        return None
    def selected_measurement_changed(self):
        self.refresh_selected_points()
        if self.measurement_view is None or self.measurement_model is None: return None
        #logger.debug("update_selected_points ({}, {})".format(self, redraw))
        row = self.measurement_view.selectionModel().selectedRows()
        # TODO: this doesn't work yet; ideally, update the plots in the Chemical Abundances tab.
        if len(row)>0:
            row = row[0]
            # We're only going to plot the first one anyway
            row_index = row.row()
            self.parent.chemical_abundances_tab.measurement_view.update_row(row_index)
            spectral_model = self.measurement_model.get_models_from_rows([row_index])[0]
            self.parent.chemical_abundances_tab.selected_model_changed(selected_model=spectral_model)

        return None

    def _init_summary_table(self):
        # Create summary model
        self.summary_model = MeasurementSummaryTableModel(self, self.parent.session)
        # Create and link summary view
        self.summary_view = BaseTableView(self)
        self.summary_view.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.summary_view.setModel(self.summary_model)
        _ = self.summary_view.selectionModel()
        _.selectionChanged.connect(self.selected_summary_changed)

    def _init_measurement_table(self):
        self.full_measurement_model = MeasurementTableModelBase(self, self.parent.session, 
                                                    ["is_acceptable",
                                                     "wavelength","expot","loggf",
                                                     "equivalent_width","reduced_equivalent_width",
                                                     "abundances","abundance_uncertainties",
                                                     "abundance_nlte","abundance_nlte_filled",
                                                     "nlte_delta","abundances_to_solar",
                                                     "is_upper_limit","user_flag"])
        self.measurement_model = MeasurementTableModelProxy(self)
        self.measurement_model.setSourceModel(self.full_measurement_model)
        #self.measurement_view = MeasurementTableView(self)
        #self.measurement_view.setModel(self.measurement_model)
        vbox, measurement_view, btn_filter, btn_refresh = base.create_measurement_table_with_buttons(
            self, self.measurement_model, self.parent.session,
            callbacks_after_menu=[self.new_session_loaded],
            display_fitting_options=False)
        self.measurement_view = measurement_view
        self.measurement_model.add_view_to_update(self.measurement_view)
        _ = self.measurement_view.selectionModel()
        _.selectionChanged.connect(self.selected_measurement_changed)
        ## Sorting crashes :(
        #self.measurement_view.setSortingEnabled(True)
        #self.measurement_view.setItemDelegate(MeasurementTableDelegate(self,self.measurement_view))
        self.measurement_view.setSizePolicy(QtGui.QSizePolicy(
            QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.MinimumExpanding))

        self.btn_filter_acceptable = btn_filter
        self.btn_filter_acceptable.clicked.connect(self.refresh_plots)
        self.btn_refresh = btn_refresh
        self.btn_refresh.clicked.connect(self.new_session_loaded)

        return vbox

    def _init_scatterplots(self):
        filters = [lambda x: (x.is_acceptable) and (not x.user_flag) and (not x.is_upper_limit),
                   lambda x: not x.is_acceptable,
                   lambda x: x.user_flag,
                   lambda x: x.is_upper_limit]
        point_styles = [{"s":30,"facecolor":"k","edgecolor":"k","alpha":0.5},
                        {"s":30,"c":"c","marker":"x","linewidths":3},
                        {"s":30,"edgecolor":"r","facecolor":"k","alpha":0.5,"linewidths":3},
                        {"s":30,"edgecolor":"r","facecolor":"none","marker":"v"}
        ]
        linefit_styles = [{"color":"k","linestyle":"--","zorder":-99},None,None,None]
        linemean_styles = [{"color":"k","linestyle":":","zorder":-999},None,None,None]
        # E. Holmbeck added error_styles:
        error_styles = [{"ms":40,"markerfacecolor":"None","markeredgecolor":"k","ecolor":"k","lw":1},
                        {"ms":40,"markerfacecolor":"None","markeredgecolor":"k","ecolor":"c","lw":1},
                        {"ms":70,"markerfacecolor":"none","markeredgecolor":"red","ecolor":"red","lw":2},
                        {"ms":0,"markerfacecolor":"none","markeredgecolor":"none","ecolor":"none"},
                        ]
        self.plot1 = SMHScatterplot(None, "expot", "abundances",# eyattr="abundance_uncertainties",
                                    tableview=self.measurement_view,
                                    filters=filters, point_styles=point_styles, error_styles=error_styles,
                                    linefit_styles=linefit_styles,linemean_styles=linemean_styles)
        self.plot2 = SMHScatterplot(None, "reduced_equivalent_width", "abundances",# eyattr="abundance_uncertainties",
                                    tableview=self.measurement_view,
                                    filters=filters, point_styles=point_styles, error_styles=error_styles,
                                    linefit_styles=linefit_styles,linemean_styles=linemean_styles)
        self.plot3 = SMHScatterplot(None, "wavelength", "abundances",# eyattr="abundance_uncertainties",
                                    tableview=self.measurement_view,
                                    filters=filters, point_styles=point_styles, error_styles=error_styles,
                                    linefit_styles=linefit_styles,linemean_styles=linemean_styles)
        
        sp = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, 
                               QtGui.QSizePolicy.MinimumExpanding)
        for fig in [self.plot1, self.plot2, self.plot3]:
            fig.setSizePolicy(sp)
        return None
        
def _model_has_species(model, species):
    for s in model.species:
        if isinstance(s, list) and species in s: return True
        elif isinstance(s, float) and species == s: return True
    return False
