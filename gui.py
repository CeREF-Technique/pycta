# -*- coding: utf-8 -*-
##########################################################################
# File name : gui.py
# Usages : Description of the module
# Start date : 05 January 2018
# Last review : 05 January 2018
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

from Tkinter import *

from config import CSV_PATH

FILES = os.listdir(CSV_PATH)

master = Tk()

label = Label(master, text="Hello World")
label.pack()

#p = PanedWindow(master, orient=VERTICAL)
#p.pack(side=TOP, expand=Y, fill=BOTH, pady=2, padx=2)
#p.add(Label(p, text='Volet 1', background='blue', anchor=CENTER))
#p.add(Label(p, text='Volet 2', background='white', anchor=CENTER) )
#p.add(Label(p, text='Volet 3', background='red', anchor=CENTER) )
#p.pack()

frame = LabelFrame(master, text="Titre de la frame", padx=20, pady=20)
frame.pack(fill="both", expand="yes")

variable = StringVar(master)
variable.set(FILES[0])

Label(frame, text="Select a file:").pack()
w = apply(OptionMenu, (frame, variable) + tuple(FILES))
w.pack()



master.mainloop()

if __name__ == '__main__':
    """
    """