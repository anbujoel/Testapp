# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:03:07 2021

@author: AJ
"""

from os import environ
from flask import Flask

app = Flask(__name__)
app.run(environ.get('PORT'))