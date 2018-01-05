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

from config import CSV_PATH

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import processing as ps
#from scipy import integrate

class CTA():
    """
    """
    def __init__(self):
        """
        """
        self.CWD = os.getcwd()
        self.visits = list()
        self.areas = list()
        
    def read_csv_folder(FOLDER_PATH):
        """
        """
        
    
    def read_csv_file(self,FILE_NAME):
        """
        Parameters
        ----------
        FILE_NAME : str
            Name of the csv file to read.
        
        Returns
        -------
            DataFrame containing all visits in the csv file.
        """
        FILE_PATH = os.path.join(CSV_PATH, FILE_NAME)
        
        try:
            self.df = pd.read_csv(FILE_PATH,header=1,names=["date","hour","ID","CH4","CO2","weigth","scale"])
        except:
            print "Impossible to read file (check that file exist and is in CSV format)"
    
    def split_visits(self):
        """
        Split the loaded CSV file in individual **Visit**
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
    
    def peaks_detect(self,delta = 0.001):
        """
        Detect peak (minimums and maximums) in waveforms.
        
        Parameters
        ----------
        delta : float
        """
        for visit in self.visits:
            visit.peak_detect(delta=delta)
    
    def compute_areas(self, data="CH4"):
        """
        Computes area under each **Visit** curve.
        
        Parameters
        ----------
        delta : float
            Step to consider a point maximum or minimum.
        """
        for visit in self.visits:
            self.areas.append(visit.compute_area(data=data))
    
    def plot_visit(self, idx):
        """
        Plot the given **Visit** curve.
        
        Parameters
        ----------
        idx : int
            Index of the **Visit** to plot.
        """
        try:
            self.visits[idx].plot_visit()
        except:
            print "Index out of range"






class Visit():
    """
    """
    def __init__(self,df):
        """
        """
        self.data = df
        self.y_CO2 = df.CO2.tolist()
        self.y_CH4 = df.CH4.tolist()
    
    def peak_detect(self,delta = 0.001):
        """
        Detect peak (minimums and maximums) in waveforms.
        
        Parameters
        ----------
        delta : float
        """
        x = np.arange(0,len(self.y_CH4),1).tolist()
        
        self.max_pk_CO2, self.min_pk_CO2 = ps.peakdetect(x,self.y_CO2,delta)
        self.max_pk_CH4, self.min_pk_CH4 = ps.peakdetect(x,self.y_CH4,delta)
    
    def compute_area(self, data="CH4"):
        """
        Computes area under each **Visit** curve.
        
        Parameters
        ----------
        delta : float
            Step to consider a point maximum or minimum.
        """
        
        if data == "CO2":
            y = self.data.CO2.tolist()
        elif data == "CH4":
            y = self.data.CH4.tolist()
        else:
            return
        
        area = np.trapz(y, dx=1.0)
        
        return area
    
    def plot_visit(self, show_peaks=False):
        """
        Plot the given **Visit** curve.
        
        Parameters
        ----------
        show_peaks : bool
            Flag set to True if maximum and minimum peak must be displayed.
        """
        
        y_CO2 = self.data.CO2.tolist()
        y_CH4 = self.data.CH4.tolist()
        
        x = np.arange(0,len(y_CH4),1).tolist()
        
        # Compute CO2 and CH4 curves
        plt.plot(x, y_CO2, 'r-', x, y_CH4, 'b-')
        
        if show_peaks == True:
            # Compute abscissa and ordinate for maximum CO2 peak values
            max_abs_CO2 = [i[0] for i in self.max_pk_CO2]
            max_ord_CO2 = [i[1] for i in self.max_pk_CO2]
            
            # Compute abscissa and ordinate for minimum CO2 peak values
            min_abs_CO2 = [i[0] for i in self.min_pk_CO2]
            min_ord_CO2 = [i[1] for i in self.min_pk_CO2]
            
            # Compute abscissa and ordinate for maximum CH4 peak values
            max_abs_CH4 = [i[0] for i in self.max_pk_CH4]
            max_ord_CH4 = [i[1] for i in self.max_pk_CH4]
            
            # Compute abscissa and ordinate for minimum CH4 peak values
            min_abs_CH4 = [i[0] for i in self.min_pk_CH4]
            min_ord_CH4 = [i[1] for i in self.min_pk_CH4]
            
            # Compute maximum CO2 peak values
            plt.plot(max_abs_CO2, max_ord_CO2,'ro')
            plt.plot(min_abs_CO2, min_ord_CO2,'rx')
            
            # Compute maximum CH4 peak values
            plt.plot(max_abs_CH4, max_ord_CH4,'bo')
            plt.plot(min_abs_CH4, min_ord_CH4,'bx')
        
        plt.show()


if __name__ == '__main__':
    
    FILE_NAME = "fichier_demo2.csv"
    
    cta = CTA()
    cta.read_csv_file(FILE_NAME)
    
    cta.split_visits()
    
    cta.compute_areas()
    
    cta.visits[0].peak_detect(0.001)
    
    cta.visits[0].plot_visit(show_peaks=True)
    
    cta.visits[0].compute_area()
