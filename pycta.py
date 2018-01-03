# -*- coding: utf-8 -*-
##########################################################################
# File name : pycta.py
# Usages : Description of the module
# Start date : 21 December 2017
# Last review : 03 January 2018
# Version : 0.1
# Author(s) : Julien Vachaudez - julien.vachaudez@cerisic.be
# License : The pycta project is distributed under the GNU General Public
#           License version 3 (https://www.gnu.org/licenses/gpl-3.0.html)
# Dependencies : Python 2.7
##########################################################################

import os
import sys

import pandas as pd


FILEPATH = r"C:\1_CERISIC\CERISIC\Projet CTA\example\fichier_demo.csv"

df = pd.read_csv(FILEPATH,header=1,names=["date","hour","ID","CH4","CO2","weigth","scale"])


class CTA():
    """
    """
    def __init__(self):
        """
        """
        visits = list()
    
    def read_csv_folder(FOLDER_PATH):
        """
        """
        
    
    def read_csv_file(FILE_PATH):
        """
        Parameters
        ----------
        FILE_PATH : str
            String containing the path to the csv file.
    
        Returns
        -------
            Dataframe containing all visits in the csv file.
        """
        df = pd.read_csv(FILEPATH,header=1,names=["date","hour","ID","CH4","CO2","weigth","scale"])
    
    def split_visits():
        """
        """
        
    
    def peaks_detect():
        """
        """
        
    
    def compute_areas():
        """
        """
        
        






class Visit(CTA):
    """
    """
    def __init__(self):
        """
        """
        
    
    def peak_detect():
        """
        """
        
    
    def compute_area():
        """
        """
        


if __name__ == '__main__':
    
