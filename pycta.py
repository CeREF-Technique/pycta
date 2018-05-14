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
        self.mock_visits = list()

    def read_csv_folder(FOLDER_PATH):
        """
        """


    def read_csv_file(self, FILE_NAME):
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


    def drop_visits(self, min_duration=MIN_VISIT_DURATION, max_duration=None, time_step=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES):
        """
            Exclude some too short or too long visits
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


                


    def mock_visit(self, nbr_to_plot):
        """
        Parameters
        ----------
        nbr_to_plot : int
            Number of visits to plot (the nth first visits)

        Returns
        -------
            A plot of the nbr_to_plot first visits.
            With some vertical separators between the visits and the ID and scale of each visit.
        """
        
        if (nbr_to_plot > len(self.visits)):
            nbr_to_plot = len(self.visits)
            
        prev_row = {'ID':'0'} 
        data_to_plot = [] #list of the indexes of the "change ID" of a visit
        for index, row in self.df.iterrows():
            #print row['ID']
            if row['ID'] != prev_row['ID']: # change of animal
               data_to_plot.append(index)
            prev_row = row

        prev = 0 # prev index
        for i in data_to_plot:
            if (i - prev) >= MIN_VISIT_DURATION/NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES: # check if the visit is long enough
                self.mock_visits.append(Visit(self.df.iloc[prev:i])) # select only the nbr_to _plot first visits
                nbr_to_plot -= 1
                if nbr_to_plot <= 0:
                    break
            prev = i

        labels = [] # contain the label of each visit to show on the graphe (ID + scale)
        limits = [] # The limit of each visit (in samples)
        y_CO2 = list() # All the data of the CO2 curve (the nth first visits)
        y_CH4 = list() # All the data of the CH4 curve (the nth first visits)

        for visit in self.mock_visits:
            y_CO2.extend(visit.data.CO2.tolist())
            y_CH4.extend(visit.data.CH4.tolist())
            labels.append(visit.data.ID.unique()[0] + " " + visit.data.scale.unique()[0])
            limits.append(len(visit.data.CO2.tolist()))

        y_CH4_CO2 = [x/y for x,y in zip(y_CH4,y_CO2)]
            
        x = np.arange(0, len(y_CO2) * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
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

        new_fixed_axisCH4CO2 = axCH4CO2.get_grid_helper().new_fixed_axis
        axCH4CO2.axis["right"] = new_fixed_axisCH4CO2(loc="right",
                                            axes=axCH4CO2,
                                            offset=(50, 0))

        p1, = axCO2.plot(x, y_CO2, 'r-', label="CO2")
        p2, = axCH4.plot(x, y_CH4, 'b-', label="CH4")
        p3, = axCH4CO2.plot(x, y_CH4_CO2, 'g-', label="CH4/CO2")

        axCO2.set_xlabel('Seconds')
        axCO2.set_ylabel("CO2")
        axCH4.set_ylabel("CH4")
        axCH4CO2.set_ylabel("CH4/CO2")

        axCO2.yaxis.label.set_color(p1.get_color())
        axCH4.yaxis.label.set_color(p2.get_color())
        axCH4CO2.yaxis.label.set_color(p3.get_color())

        curr_time = 0
        y_pos = axCO2.get_ylim() # [bottom, top]
        y_pos = y_pos[0] + 0.93 * (y_pos[1] - y_pos[0]) # set the text at 93 % of the height
        for i in range(len(limits)):
            curr_time += limits[i] * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES
            plt.axvline(x=curr_time, linewidth=0.5, color='#555555')
            axCO2.text(curr_time - 50, y_pos, labels[i], ha="right", va="center", size=8, rotation=90)
        
        areaCO2 = np.trapz(y_CO2, dx=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES)
        areaCH4 = np.trapz(y_CH4, dx=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES)
        areaCH4_CO2 = np.trapz(y_CH4_CO2, dx=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES)
        plt.title("Areas :    CO2 : %.3f    CH4 : %.3f    CH4/CO2 : %.3f" %
                (areaCO2, areaCH4, areaCH4_CO2))

        plt.subplots_adjust(left=0.03,bottom=0.05, right=0.92,top=0.96) # used to export the graphs in PNG on a big screen (24")
        plt.draw()
        plt.show()

    
    def num_visits(self):
        """
        Returns
        -------
            Prints the number of visits
        """
        print len(self.visits)


    def peaks_detect(self, delta=0.001):
        """
        Detect peak (minimums and maximums) in waveforms.

        Parameters
        ----------
            delta : float
            Used to detect the minimums and maximums
        """
        for visit in self.visits:
            visit.peak_detect(delta=delta)


    def plot_visit(self, idx, show_peaks=False, show_areas=True):
        """
        Plot the given **Visit** curve.

        Parameters
        ----------
            idx : int
            Index of the **Visit** to plot.
            
        """
        if idx >= 0 and idx < len(self.visits):
            #Check that the user enters a valid index
            self.visits[idx].plot_visit(show_peaks=show_peaks,
                                        show_areas=show_areas)
        else:
            print "Index out of range"
    def export_results(self):
        """
            Export the result to a csv file.
            1 line = one visit
        """






class Visit():
    """
    Descibes a visit.
    Has a data (Pandas.DataFrame)
    y_CO2 = list of the CO2 amplitudes
    y_CH4 = list of the CH4 amplitudes
    y_CH4_CO2 = list of the CH4/CO2 amplitudes
    ID = ID of the cow
    scale = ID of the used scale
    area_CO2 = area of the CO2 curve
    area_CH4 = area of the CH4 curve
    area_CH4_CO2 = area of the CH4_CO2 curve
    """
    def __init__(self, df):
        """
            df : dataframe (from panda)
            dataframe for one visit
        """
        self.data = df
        self.y_CO2 = df.CO2.tolist()
        self.y_CH4 = df.CH4.tolist()
        self.ID = df.ID.unique()[0]
        self.scale = df.scale.unique()[0]
        self.len_CO2 = len(self.y_CO2)
        self.len_CH4 = len(self.y_CH4)
        
        self.delete_begin_visit()
        self.clip_data()
        if self.len_CO2 == self.len_CH4:
            self.y_CH4_CO2 = [x/y for x,y in zip(self.y_CH4,self.y_CO2)]
        elif self.len_CO2 > self.len_CH4:
            self.y_CH4_CO2 = [x/y for x,y in zip(self.y_CH4,self.y_CO2[-self.len_CH4:])] # take the last elements for the CO2 array which is greater than the CH4 array
        else : # self.len_CO2 < len(self.y_CH4)
            self.y_CH4_CO2 = [x/y for x,y in zip(self.y_CH4[-self.len_CO2:],self.y_CO2)] # take the last elements for the CH4 array which is greater than the CO2 array       
        
        # areas of each curve
        self.area_CO2 = 0.0
        self.area_CH4 = 0.0
        self.area_CH4_CO2 = 0.0
        self.compute_area()
        

    def peak_detect(self, delta=0.001):
        """
        Detect peak (minimums and maximums) in waveforms.

        Parameters
        ----------
            delta : float
        """

        if self.len_CH4 == self.len_CO2:
            x_CH4 = np.arange(0, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CO2 = np.arange(0, self.len_CO2 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CH4CO2 = np.arange(0, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            
        elif self.len_CO2 > self.len_CH4:
            start_time = (self.len_CO2 - self.len_CH4) * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES
            x_CH4 = np.arange(start_time, self.len_CO2 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CO2 = np.arange(0, self.len_CO2 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CH4CO2 = np.arange(start_time, self.len_CO2 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            
        else: # self.len_CO2 < self.len_CH4
            start_time = (self.len_CH4 - self.len_CO2) * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES
            x_CH4 = np.arange(0, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CO2 = np.arange(start_time, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CH4CO2 = np.arange(start_time, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()

        self.max_pk_CO2, self.min_pk_CO2 = ps.peakdetect(x_CO2, self.y_CO2, delta)
        self.max_pk_CH4, self.min_pk_CH4 = ps.peakdetect(x_CH4 ,self.y_CH4, delta)
        self.max_pk_CH4_CO2, self.min_pk_CH4_CO2 = ps.peakdetect(x_CH4CO2, self.y_CH4_CO2, delta)


    def delete_begin_visit(self, delete_duration_co2=DELETE_SECONDS_CO2, delete_duration_ch4=DELETE_SECONDS_CH4):
        """
        Parameters
        ----------
        delete_duration_co2 : int
            Number of seconds to delete at the begining of the visit for the CO2
        delete_duration_ch4 : int
             Number of seconds to delete at the begining of the visit for the CH4
        """
        nbr_of_pts_to_delete_co2 = int(np.trunc(delete_duration_co2 / NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES))
        nbr_of_pts_to_delete_ch4 = int(np.trunc(delete_duration_ch4 / NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES))
        min_samples = int(np.trunc(MIN_VISIT_DURATION / NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES))

        if self.len_CO2 > min_samples: # First check if the visit will not be deleted when filtering
            if self.len_CO2 > nbr_of_pts_to_delete_co2: # Second, chek if there is enough points to delete
                for i in range(nbr_of_pts_to_delete_co2):
                    del self.y_CO2[i]
                self.len_CO2 = len(self.y_CO2) # update the CO2 lenght
            else:
                 print "not enough points to delete a the begining of the visit for the ID : ", self.ID ," in the CO2 samples"      
            if self.len_CH4 > nbr_of_pts_to_delete_ch4:
                for i in range(nbr_of_pts_to_delete_ch4):
                    del self.y_CH4[i]
                self.len_CH4 = len(self.y_CH4) # update the CH4 lenght
            else:
                print "not enough points to delete a the begining of the visit for the ID : ", self.ID," in the CH4 samples"
        
        

    def clip_data(self):
        """
            Filters all the data of the visit that doesn't fit inside the max and min values of each curve
            The values are clipped to the max or the min

            NB : MIN and MAX values are defined into the conf.py file
        """
        data_array = np.arange(self.len_CO2) # initialize the data_array to the right length
        data_array = np.clip(self.y_CO2, MIN_CO2, MAX_CO2)
        self.y_CO2 = data_array.tolist()

        data_array = np.arange(self.len_CH4) # initialize the data_array to the right length
        data_array = np.clip(self.y_CH4, MIN_CH4, MAX_CH4)
        self.y_CH4 = data_array.tolist()


    def compute_area(self):
        """
        Computes area under each **Visit** curve.
        The area is calculated with the trapeze method
        Results are stored into area_CO2, area_CH4 and area_CH4_CO2
        """
        self.area_CO2 = np.trapz(self.y_CO2, dx=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES)
        self.area_CH4 = np.trapz(self.y_CH4, dx=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES)
        self.area_CH4_CO2 = np.trapz(self.y_CH4_CO2, dx=NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES)


    def plot_visit(self, show_peaks=False, show_areas=True, show_ID=False):
        """
        Plot the given **Visit** curve.

        Parameters
        ----------
        show_peaks : bool
            Flag set to True if maximum and minimum peak must be displayed.
        show_areas : bool
            Flag set to True to show the calculation of the areas in the title
        show_ID : bool
            Flag set to True to show the ID of the visit(s)
        """
        if self.len_CH4 == self.len_CO2:
            x_CH4 = np.arange(0, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CO2 = np.arange(0, self.len_CO2 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CH4CO2 = np.arange(0, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            
        elif self.len_CO2 > self.len_CH4:
            start_time = (self.len_CO2 - self.len_CH4) * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES
            x_CH4 = np.arange(start_time, self.len_CO2 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CO2 = np.arange(0, self.len_CO2 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CH4CO2 = np.arange(start_time, self.len_CO2 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            
        else: # self.len_CO2 < self.len_CH4
            start_time = (self.len_CH4 - self.len_CO2) * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES
            x_CH4 = np.arange(0, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CO2 = np.arange(start_time, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
                          NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES).tolist()
            x_CH4CO2 = np.arange(start_time, self.len_CH4 * NBR_OF_SECONDS_BETWEEN_TWO_SAMPLES,
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

        new_fixed_axisCH4CO2 = axCH4CO2.get_grid_helper().new_fixed_axis
        axCH4CO2.axis["right"] = new_fixed_axisCH4CO2(loc="right",
                                            axes=axCH4CO2,
                                            offset=(50, 0))


        p1, = axCO2.plot(x_CO2, self.y_CO2, 'r-', label="CO2")
        p2, = axCH4.plot(x_CH4, self.y_CH4, 'b-', label="CH4")
        p3, = axCH4CO2.plot(x_CH4CO2, self.y_CH4_CO2, 'g-', label="CH4/CO2")


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
            axCO2.plot(max_abs_CO2, max_ord_CO2, 'ro')
            axCO2.plot(min_abs_CO2, min_ord_CO2, 'rx')

            # Plot maximum CH4 peak values
            axCH4.plot(max_abs_CH4, max_ord_CH4, 'bo')
            axCH4.plot(min_abs_CH4, min_ord_CH4, 'bx')

            # Plot maximum CH4/CO2 peak values
            axCH4CO2.plot(max_abs_CH4_CO2, max_ord_CH4_CO2, 'go')
            axCH4CO2.plot(min_abs_CH4_CO2, min_ord_CH4_CO2, 'gx')

        if show_areas:
            plt.title("Areas :    CO2 : %.3f    CH4 : %.3f    CH4/CO2 : %.3f" %
                  (self.area_CO2, self.area_CH4, self.area_CH4_CO2))

        if show_ID:
            x_pos = x[len(x)/2]
            y_pos = axCO2.get_ylim() # [bottom, top]
            y_pos = y_pos[0] + 0.97*(y_pos[1] - y_pos[0]) # set the text at 97 % of the height
            axCO2.text(x_pos, y_pos, self.ID[0], ha="center", va="center", size=8)
                    

        #plt.subplots_adjust(left=0.03,bottom=0.05, right=0.92,top=0.96) # used to export the graphs in PNG on a big screen (24")
        plt.draw()
        plt.show()



if __name__ == '__main__':

    #FILE_NAME = "fichier_demo2.csv"
    FILE_NAME = "exportFermeCTA_4_5_17.csv"

    cta = CTA()
    cta.read_csv_file(FILE_NAME)

    cta.split_visits()

    cta.drop_visits()
    #cta.mock_visit(20)

    cta.peaks_detect(delta=0.001)
    cta.plot_visit(0,show_peaks=True)
    cta.plot_visit(1,show_peaks=True)
    cta.plot_visit(2,show_peaks=True)
    cta.plot_visit(3,show_peaks=True)
#    cta.visits[0].plot_visit(show_peaks=True)

