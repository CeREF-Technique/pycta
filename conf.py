# -*- coding: utf-8 -*-
##########################################################################
# File name : config.py
# Usages : Description of the module
# Start date : Thu Jan 04 13:09:58 2018
# Last review : Thu Jan 04 13:09:58 2018
# Version : 0.1
# Author(s) : Julien Vachaudez - julien.vachaudez@cerisic.be
# License : The pycta project is distributed under the GNU General Public
#           License version 3 (https://www.gnu.org/licenses/gpl-3.0.html)
# Dependencies : Python 2.7
##########################################################################

import os

# Constants for configuration

CWD = os.getcwd()
CSV_PATH = os.path.join(CWD, "csv")
NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES = 3.0

# Maximums and minimums for each curve
MAX_CO2 = 1.25
MIN_CO2 = 0.25

MAX_CH4 = 0.11
MIN_CH4 = 0.02

MAX_CH4_CO2 = 0.5
MIN_CH4_CO2 = 0.1
