import numpy as np
import os
np.random.seed(0)

from bokeh.io import curdoc,output_file,save
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.models import Spacer, ColumnDataSource, Select, Slider, TextInput
from bokeh.models import HoverTool, SaveTool, ResetTool, ZoomInTool, ZoomOutTool, WheelZoomTool, BoxZoomTool
from bokeh.plotting import figure, show
from bokeh.palettes import Spectral6

from bokeh.models.widgets import RangeSlider, RadioGroup, CheckboxGroup, Button, Div, Paragraph, Panel, Tabs
from bokeh.models.widgets.inputs import DatePicker

from bokeh.embed import file_html


import pandas as pd
from datetime import datetime

CSV_PATH = r"C:\1_CERISIC\CERISIC\Projet CTA\csv"

import pyautogui
(page_width, page_height) = pyautogui.size()
margin = 0.1
page_width = int((1-2*margin)*page_width)
del pyautogui

FILEPATH = r"C:\1_CERISIC\CERISIC\Projet CTA\example\fichier_demo.csv"

df = pd.read_csv(FILEPATH,header=1,names=["date","hour","ID","CH4","CO2","weigth","scale"])

dates  = df.date.unique().tolist()
IDs    = df.ID.unique().tolist()
scales = df.scale.unique().tolist()


# Create indicators list

file_list = os.listdir(CSV_PATH)


# Group data by ID
grp = df.reset_index().groupby('ID')['index'].apply(np.array)

##############################
# Define callbacks functions #
##############################


#def file_select_handler():
#    print("Previous label: ")
#    print("Updated label: ")

#def checkbox_IDs_handler(attr, old, new):
#    print("Previous label: " + old)
#    print("Updated label: " + new)
#
#def checkbox_scales_handler(attr, old, new):
#    print("Previous label: " + old)
#    print("Updated label: " + new)




#########################
# Create Input controls #
#########################

# Create Dataset Div and RadioGroups Widget
dataset_div = Div(text="<font size='4'><b>Application</b></font>", width=page_width)


file_select = Select(value='Select File',
                         title='Select File:',
                         width=page_width/4,
                         options=file_list)


options_div = Div(text="<font size='4'><b>Options</b></font>", width=page_width)
#
#max_date = datetime(int(max(dates).split("-")[0]),int(max(dates).split("-")[1]),int(max(dates).split("-")[2]))
#min_date = datetime(int(min(dates).split("-")[0]),int(min(dates).split("-")[1]),int(min(dates).split("-")[2]))
#
#
#datepicker = DatePicker(title="Date", min_date=min_date,
#                       max_date=max_date,
#                       value=datetime(datetime.now().year,1,1)
#                       )
#


checkbox_IDs = CheckboxGroup(
        labels=IDs, active=[0])

checkbox_scales = CheckboxGroup(
        labels=scales, active=[0])



range_slider = RangeSlider(start=0, end=10, value=(1,9), step=.1, title="Stuff")

#
#
#plot_title = TextInput(title="Title :", value='')

source = ColumnDataSource(data=dict(x=[], y=[]))

source.data = dict(
    y = df[df['ID'] == "BE156915425"].ix[:, 'CH4'].tolist(),
    x = range(len(y))
)


#hover = HoverTool(tooltips=[
#    ("Customer", "@customer"),
#    ("Area", "@area"),
#    ("Equipment", "@equipment"),
#    ("POM", "@POM"),
#    ("x","@x"),
#    ("y","@y")
#])
#
p = figure(plot_width=page_width, plot_height=page_width/2, title="", toolbar_location="right", tools=[SaveTool(),ResetTool(),ZoomInTool(),ZoomOutTool(),WheelZoomTool(),BoxZoomTool()])
p.min_border_left = 80
p.min_border_bottom = 80
p.background_fill_color = "white"
##p.background_fill_alpha = 0


#p.circle(x=X, y=Y, size=10)
p.circle(x="x", y="y",source=source, size=5, color="navy", alpha=0.5)
#X = x=df[df['ID'] == "BE156915425"].ix[:, 'date'].tolist()
#Y=df[df['ID'] == "BE156915425"].ix[:, 'CH4'].tolist()
#p.circle(x=X, y=Y, source=source, size=10, color="color", alpha=0.5)
#
#
#update_button = Button(label="Update Plot", button_type="success")
#


#file_select.on_change("value", file_select_handler)
#checkbox_IDs.on_change("value", checkbox_IDs_handler)
#checkbox_scales.on_change("value", checkbox_scales_handler)



#################
# Set up layout #
#################

lay = column(dataset_div,
             row(file_select),
             options_div,
             row(checkbox_IDs,checkbox_scales),
             row(p)
             )


curdoc().add_root(lay)
curdoc().title = "cta"