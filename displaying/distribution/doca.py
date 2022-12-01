import sys
import numpy as np
from scipy.stats import linregress
from scipy.stats import t
from scipy.stats import lognorm
from scipy.stats import gamma
from sklearn.utils import resample as bootstrap
import matplotlib.pyplot as plt
from os.path import join, abspath, dirname

currentdir = dirname(abspath(__file__))
sys.path.append(join(currentdir,'../../processing'))

import processing.gmst as gmst
import processing.lens as lens
import processing.utils as ut
import processing.series as se

lens2_gmst_full = gmst.get_gmst_annual_lens2_ensmean()
lens2_prec_l1_pat_full = lens.get_LENS1_JFM_precip_NOAA_PM_NN()
lens2_prec_l1_cen_full = lens.get_LENS1_annual_precip_NOAA_QN_NN()
lens2_prec_l2_pat_full = lens.get_LENS2_JFM_precip_NOAA_PM_NN()
lens2_prec_l2_cen_full = lens.get_LENS2_annual_precip_NOAA_QN_NN()

lens2_prec_l1_pat_mean = lens2_prec_l1_pat_full.sel(time=slice('1991', '2020')).mean(['time', 'run'])
lens2_prec_l2_pat_mean = lens2_prec_l2_pat_full.sel(time=slice('1991', '2020')).mean(['time', 'run'])
lens2_prec_l1_cen_mean = lens2_prec_l1_cen_full.sel(time=slice('1991', '2020')).mean(['time', 'run'])
lens2_prec_l2_cen_mean = lens2_prec_l2_cen_full.sel(time=slice('1991', '2020')).mean(['time', 'run'])