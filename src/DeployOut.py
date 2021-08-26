import os
import pandas as pd
import numpy as np


def get_csv_output(model, data):
    file_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/output_result/" + str(model) + ".csv"
    data.to_csv(file_name, header=True)
