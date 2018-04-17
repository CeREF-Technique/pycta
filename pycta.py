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

from conf import * # CSV_PATH, NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES, ...

import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

import processing as ps


class CTA():
    """
    """
    def __init__(self):
        """
        """
        self.visits = list()
        self.areas = list()
        self.mock_visits = list()
        
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
    
    def drop_visits(self,min_duration=180,max_duration=None,time_step=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES):
        """
            Exclude some too short or too  long visits
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
        for i in range(len(to_drop)-1,-1,-1):
            self.visits.pop(to_drop[i])
    
    def mock_visit(self,start,stop,step=1):
        """
        """
        if (stop > len(self.visits)):
            stop = len(self.visits)

        mock_dataframe = self.visits[start].data
        for i in range(start+step,stop,step):
            mock_dataframe = mock_dataframe.append(self.visits[i].data)

        #print mock_dataframe
        mock_visit = Visit(mock_dataframe)
        mock_visit.filter_data()
        mock_visit.plot_visit(show_areas=True)
    
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
    
    def plot_visit(self, idx, show_peaks=False,show_areas=True):
        """
        Plot the given **Visit** curve.
        
        Parameters
        ----------
        idx : int
            Index of the **Visit** to plot.
        """
        if idx >=0 and idx < len(self.visits):
            #Check that the user enters a valid index
            self.visits[idx].plot_visit(show_peaks=show_peaks,
                                        show_areas=show_areas)
        else:
            print "Index out of range"






class Visit():
    """
    """
    def __init__(self, df):
        """
            df : dataframe (from panda)
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


    def filter_data(self):
        """
            Filters all the data of the visit that doesn't fit inside the max and min values of each curve
            The values are clipped to the max or the min
        """
        data_array = np.arange(len(self.y_CO2))
        data_array = np.clip(self.y_CO2, MIN_CO2, MAX_CO2)
        self.y_CO2 = data_array.tolist()

        data_array = np.clip(self.y_CH4, MIN_CH4, MAX_CH4)
        self.y_CH4 = data_array.tolist()

        data_array = np.clip(self.y_CH4_CO2, MIN_CH4_CO2, MAX_CH4_CO2)
        self.y_CH4_CO2 = data_array.tolist()
        
    
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
        
        area = np.trapz(y, dx=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES)
        
        return area
    
    def plot_visit(self, show_peaks=False, show_areas=True):
        """
        Plot the given **Visit** curve.
        
        Parameters
        ----------
        show_peaks : bool
            Flag set to True if maximum and minimum peak must be displayed.
        """
        
        x = np.arange(0,len(self.y_CH4)*NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                      NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
        
        
        axCO2 = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)
        axCO2.autoscale()
        
        axCH4 = axCO2.twinx()
        axCH4CO2 = axCO2.twinx()
        
        new_fixed_axisCH4 = axCH4.get_grid_helper().new_fixed_axis
        axCH4.axis["right"] = new_fixed_axisCH4(loc="right",
                                            axes=axCH4,
                                            offset=(0, 0))
        
        new_fixed_axis3 = axCH4CO2.get_grid_helper().new_fixed_axis
        axCH4CO2.axis["right"] = new_fixed_axis3(loc="right",
                                            axes=axCH4CO2,
                                            offset=(50, 0))
        
        
        p1, = axCO2.plot(x,self.y_CO2,'r-',label="CO2")
        p2, = axCH4.plot(x,self.y_CH4,'b-',label="CH4")
        p3, = axCH4CO2.plot(x,self.y_CH4_CO2,'g-',label="CH4/CO2")
        
        
        axCO2.set_xlabel('Seconds')
        axCO2.set_ylabel("CO2")
        axCH4.set_ylabel("CH4")
        axCH4CO2.set_ylabel("CH4/CO2")
        
        
        axCO2.yaxis.label.set_color(p1.get_color())
        axCH4.yaxis.label.set_color(p2.get_color())
        axCH4CO2.yaxis.label.set_color(p3.get_color())
        
        
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
            axCO2.plot(max_abs_CO2, max_ord_CO2,'ro')
            axCO2.plot(min_abs_CO2, min_ord_CO2,'rx')
            
            # Plot maximum CH4 peak values
            axCH4.plot(max_abs_CH4, max_ord_CH4,'bo')
            axCH4.plot(min_abs_CH4, min_ord_CH4,'bx')
            
            # Plot maximum CH4/CO2 peak values
            axCH4CO2.plot(max_abs_CH4_CO2, max_ord_CH4_CO2,'go')
            axCH4CO2.plot(min_abs_CH4_CO2, min_ord_CH4_CO2,'gx')

        if show_areas:
            plt.title("Areas :    CO2 : %.3f    CH4 : %.3f    CH4/CO2 : %.3f" %
                  (self.compute_area("CO2"), self.compute_area("CH4"),
                   self.compute_area("CH4/CO2"))) 

        #plt.subplots_adjust(left=0.03,bottom=0.05, right=0.92,top=0.96) # used to export the graphs in PNG on a big screen (24")
        plt.draw()
        plt.show()



if __name__ == '__main__':
    
    #FILE_NAME = "fichier_demo2.csv"
    FILE_NAME = "exportFermeCTA_30_4_17.csv"
    
    cta = CTA()
    cta.read_csv_file(FILE_NAME)
    
    cta.split_visits()
    
    cta.compute_areas()
    cta.drop_visits()
    cta.mock_visit(0,20)
    
    """cta.peaks_detect(delta=0.001)
    cta.plot_visit(0,show_peaks=True)
    cta.plot_visit(1,show_peaks=True)
    cta.plot_visit(2,show_peaks=True)
    cta.plot_visit(3,show_peaks=True)"""
#    cta.visits[0].plot_visit(show_peaks=True)
#    cta.visits[0].compute_area()
