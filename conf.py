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

def get_csv_path():
    """
    """
    CWD = os.getcwd()
    CSV_PATH = os.path.join(CWD,"csv")
    
    return CSV_PATH

CWD = os.getcwd()
CSV_PATH = os.path.join(CWD,"csv")
CSV_PATH = r'C:\1_CERISIC\Git\pycta\csv'