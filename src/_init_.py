# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:54:43 2019

@author: higupta
"""

class Semantics(object):
    """
    Returns a `````` object with given name.

    """
    def __init__(self, name):
        self.name = name

    def get_details(self):
        "Returns a string containing name of the person"
        return self.name
