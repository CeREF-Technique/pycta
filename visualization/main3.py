# -*- coding: utf-8 -*-
##########################################################################
# File name : NAME.py
# Usages : Description of the module
# Start date : Thu Dec 07 22:09:32 2017
# Last review : Thu Dec 07 22:09:32 2017
# Version : 0.1
# Author(s) : Julien Vachaudez - julien.vachaudez@cerisic.be
# License : The pyautodiag project is distributed under the GNU Affero General Public 
#           License version 3 (https://www.gnu.org/licenses/agpl-3.0.html) and is also 
#           available under alternative licenses negotiated directly
#           with CERISIC.
# Dependencies : Python 2.7
##########################################################################

import os
import sys

from bokeh.plotting import curdoc, output_file
from bokeh.models.widgets import Button, CheckboxGroup
from bokeh.layouts import widgetbox, row
from bokeh.models import ColumnDataSource, Callback

output_file("states.html", title="states")

states_list = ["Alabama", "Alaska ", "Arizona", "Arkansas", "California", \
        "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", \
        "Hawaii", "Idaho ", "Illinois", "Indiana", "Iowa", "Kansas", \
        "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", \
        "Michigan ", "Minnesota", "Mississippi", "Missouri", "Montana",\
        "Nebraska", "Nevada ", "New Hampshire", "New Jersey",\
        "New Mexico", "New York", "North Carolina", "North Dakota", \
        "Ohio", "Oklahoma","Oregon", "Pennsylvania", "Rhode Island", \
        "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",\
        "Vermont", "Virginia","Washington", "West Virginia", \
        "Wisconsin", "Wyoming"]

states = CheckboxGroup(
        labels = states_list,
        active=[0,1])

select_all = Button(label="select all")

def update():
    states.active = list(range(len(states_list)))
select_all.on_click(update)

group = widgetbox(select_all, states)

layout = row(group)

curdoc().add_root(layout)
curdoc().title = "states"
