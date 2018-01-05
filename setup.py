# -*- coding: utf-8 -*-
##########################################################################
# File name : setup.py
# Usages : Description of the module
# Start date : 05 January 2018
# Last review : 05 January 2018
# Version : 0.1
# Author(s) : Julien Vachaudez - julien.vachaudez@cerisic.be
# License : The pycta project is distributed under the GNU General Public
#           License version 3 (https://www.gnu.org/licenses/gpl-3.0.html)
# Dependencies : Python 2.7
##########################################################################

from cx_Freeze import setup, Executable

build_exe_options = {
"includes": ['numpy', 'pandas'],
"packages": [],
'excludes': ['collections.abc'],
#'excludes' : ['boto.compat.sys',
#              'boto.compat._sre',
#              'boto.compat._json',
#              'boto.compat._locale',
#              'boto.compat._struct',
#              'boto.compat.array'],
"include_files": []}

setup(
    name = "pycta",
    version = "0.1",
    description = "Determine the right diet by following the CO2 and CH4 levels emitted by livestock.",
    author = "CERISIC",
    options = {"build_exe": build_exe_options},
    executables = [Executable("pycta.py")]
)