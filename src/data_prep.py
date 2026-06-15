import numpy as np

import pandas as pd


def convert_to_m(df):
    condition = df["unit_of_measurement"] == "ft"
    df.loc[condition, "altitude_mean_meters"] = df.loc[condition, "altitude_mean_meters"] * 0.3048
    df.loc[condition,"unit_of_measurement"] = 'm'
    return df