import numpy as np
import pandas as pd
from scipy import signal
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

class Data():

    def __init__(self, path):
        self.path = path