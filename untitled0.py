# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 13:08:26 2025

@author: igor
"""

my_list = [[13,2], [1,2], [2,2]]
element = [1,2]

try:
    index = my_list.index(element)
    print(f"Found {element} at index {index}.")
except ValueError:
    print(f"{element} is not in the list.")
