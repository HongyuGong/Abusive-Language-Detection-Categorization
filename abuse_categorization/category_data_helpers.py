"""
Data helpers for abusive language categorization
"""

import sys
import os
import csv
import pickle
import numpy as np
import param


def loadCategoryData(data_type, verbose=True):
    assert train_type in ["train", "test"]
    
