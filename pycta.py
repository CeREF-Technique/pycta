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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import integrate

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
        visits_index = list()
        
        # Group DataFrame by IDs
        grp = self.df.groupby('ID')
        
        # List all unique IDs
        IDs = self.df.ID.unique().tolist()
        
        for ID in IDs:
            prev = 0
            
            # Select a group
            unique_grp = grp.get_group(ID)
            
            # Convert DataFrame to Numpy array
            index_array = unique_grp.index.values
            
            # Split array in visits based on index consecutive indices
            splits = np.append(np.where(np.diff(index_array) != 1)[0],len(index_array)+1)+1
            
            #Split data n visits
            for split in splits:
                visits_index.append(index_array[prev:split])
                prev = split
        
        # Convert visits indices with corresponding data
        for visit in visits_index:
            self.visits.append(Visit(cta.df.loc[visit.astype(np.int32).tolist()]))
    
    def peaks_detect(self):
        """
        """
        
    
    def compute_areas(self):
        """
        """
        
        






class Visit(CTA):
    """
    """
    def __init__(self,df):
        """
        """
        self.data = df
    
    def peak_detect(self):
        """
        """
        
    
    def compute_area(self, data="CH4", plot=False):
        """
        """
        if data == "CO2":
            y = self.data.CO2.tolist()
        elif data == "CH4":
            y = self.data.CH4.tolist()
        else:
            return
        
        y_int = integrate.cumtrapz(y, x=None,dx=1.0, initial=0)
        
        if plot == True:
            x = np.arange(0,len(y_int),1)
            
            plt.plot(x, y_int, 'ro')
            plt.show()


if __name__ == '__main__':
    
    FILE_PATH = r"csv\fichier_demo2.csv"
    
    cta = CTA()
    cta.read_csv_file(FILE_PATH)
    
    cta.split_visits()
    
    cta.visits[0].compute_area(plot=True)
