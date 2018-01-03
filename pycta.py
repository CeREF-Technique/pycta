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
        self.CWD = os.getcwd()
        self.visits = list()
    
    def read_csv_folder(FOLDER_PATH):
        """
        """
        
    
    def read_csv_file(self,FILE_PATH):
        """
        Parameters
        ----------
        FILE_PATH : str
            String containing the relative path to the csv file.
    
        Returns
        -------
            Dataframe containing all visits in the csv file.
        """
        FILEPATH = os.path.join(self.CWD, FILE_PATH)
        
        try:
            self.df = pd.read_csv(FILEPATH,header=1,names=["date","hour","ID","CH4","CO2","weigth","scale"])
        except:
            print "Impossible to read file (check that file exist and is in CSV format)"
    
    def split_visits(self):
        """
        """
        
    
    def peaks_detect(self):
        """
        """
        
    
    def compute_areas(self):
        """
        """
        
        






class Visit(CTA):
    """
    """
    def __init__(self):
        """
        """
        
    
    def peak_detect(self):
        """
        """
        
    
    def compute_area(self):
        """
        """
        


if __name__ == '__main__':
    
    FILE_PATH = r"csv\fichier_demo"
    
    cta = CTA()
    cta.read_csv_file(FILE_PATH)