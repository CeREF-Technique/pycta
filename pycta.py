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

from conf import CSV_PATH, NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES

import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
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
    
    def drop_visits(self,min_duration=180,max_duration=None,time_step=3):
        """
        """
        to_drop = []
        
        for visit in self.visits:
            # Compute visit duration
            visit_duration = len(visit.y_CH4)*time_step
            
            # If visit duration is smaller than minimal duration
            if (visit_duration < min_duration):
                to_drop.append(self.visits.index(visit))
            if ((visit_duration > max_duration) and (max_duration != None)):
                to_drop.append(self.visits.index(visit))
        
        # Drop visits
        for i in range(len(to_drop),0,-1):
            self.visits.pop(to_drop[i])
    
    def mock_visit(self,start,stop,step=1):
        """
        """
        if (stop > len(self.visits)):
            stop = len(self.visits)
        
        for i in range(start,stop,step):
            tmp = self.visits[0]
            
            self.mock_visits.append(tmp)
    
    def num_visits(self):
        """
        """
        print len(self.visits)
    
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
    
    def plot_visit(self, idx, show_peaks=False):
        """
        Plot the given **Visit** curve.
        
        Parameters
        ----------
        idx : int
            Index of the **Visit** to plot.
        """
        try:
            self.visits[idx].plot_visit(show_peaks=show_peaks)
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
        self.y_CH4_CO2 = [x/y for x,y in zip(self.y_CH4,self.y_CO2)]
    
    def peak_detect(self,delta = 0.001):
        """
        Detect peak (minimums and maximums) in waveforms.
        
        Parameters
        ----------
        delta : float
        """
        x = np.arange(0,len(self.y_CH4)*NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                      NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
        
        self.max_pk_CO2, self.min_pk_CO2 = ps.peakdetect(x,self.y_CO2,delta)
        self.max_pk_CH4, self.min_pk_CH4 = ps.peakdetect(x,self.y_CH4,delta)
        self.max_pk_CH4_CO2, self.min_pk_CH4_CO2 = ps.peakdetect(x,self.y_CH4_CO2,delta)
    
    def compute_area(self, data="CH4"):
        """
        Computes area under each **Visit** curve.
        
        Parameters
        ----------
        delta : float
            Step to consider a point maximum or minimum.
        """
        
        if data == "CO2":
            y = self.y_CO2
        elif data == "CH4":
            y = self.y_CH4
        elif data == "CH4/CO2":
            y = self.y_CH4_CO2
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
        
        x = np.arange(0,len(self.y_CH4)*NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                      NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
        
        
        ax1 = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)
        
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        new_fixed_axis2 = ax2.get_grid_helper().new_fixed_axis
        ax2.axis["right"] = new_fixed_axis2(loc="right",
                                            axes=ax2,
                                            offset=(0, 0))
        
        new_fixed_axis3 = ax3.get_grid_helper().new_fixed_axis
        ax3.axis["right"] = new_fixed_axis3(loc="right",
                                            axes=ax3,
                                            offset=(50, 0))
        
        #ax3.axis["right"].toggle(all=True)
        
        
        p1, = ax1.plot(x,self.y_CH4,'b-',label="CH4")
        p2, = ax2.plot(x,self.y_CO2,'r-',label="CO2")
        p3, = ax3.plot(x,self.y_CH4_CO2,'g-',label="CH4/CO2")
        
        
        ax1.set_xlabel('Seconds')
        ax1.set_ylabel("CH4")
        ax2.set_ylabel("CO2")
        ax3.set_ylabel("CH4/CO2")
        
        
        ax1.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())
        ax3.yaxis.label.set_color(p3.get_color())
        
        
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
            
            # Compute abscissa and ordinate for maximum CH4 peak values
            max_abs_CH4_CO2 = [i[0] for i in self.max_pk_CH4_CO2]
            max_ord_CH4_CO2 = [i[1] for i in self.max_pk_CH4_CO2]
            
            # Compute abscissa and ordinate for minimum CH4 peak values
            min_abs_CH4_CO2 = [i[0] for i in self.min_pk_CH4_CO2]
            min_ord_CH4_CO2 = [i[1] for i in self.min_pk_CH4_CO2]
            
            
            # Plot maximum CO2 peak values
            ax1.plot(max_abs_CO2, max_ord_CO2,'ro')
            ax1.plot(min_abs_CO2, min_ord_CO2,'rx')
            
            # Plot maximum CH4 peak values
            ax2.plot(max_abs_CH4, max_ord_CH4,'bo')
            ax2.plot(min_abs_CH4, min_ord_CH4,'bx')
            
            # Plot maximum CH4/CO2 peak values
            ax3.plot(max_abs_CH4_CO2, max_ord_CH4_CO2,'go')
            ax3.plot(min_abs_CH4_CO2, min_ord_CH4_CO2,'gx')
        
        
        plt.draw()
        plt.show()



if __name__ == '__main__':
    
    #FILE_NAME = "fichier_demo2.csv"
    FILE_NAME = "exportFermeCTA_4_5_17.csv"
    
    cta = CTA()
    cta.read_csv_file(FILE_NAME)
    
    cta.split_visits()
    
    cta.compute_areas()
    
    cta.peaks_detect(delta=0.001)
    cta.plot_visit(0,show_peaks=True)
    cta.plot_visit(1,show_peaks=True)
    cta.plot_visit(2,show_peaks=True)
    cta.plot_visit(3,show_peaks=True)
    
#    cta.visits[0].plot_visit(show_peaks=True)
#    cta.visits[0].compute_area()
