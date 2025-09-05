# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Intro
#
# Some basic preprocessing to ensure required features exist in the dataframe

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

import pickle
from datetime import date

# %% [markdown]
# ### Setup

# %% [raw]
# Note: The ds_suffix is used for managing mutliple experiments. All created files will have the ds_suffix. Make sure the ds_suffix is the same in all notebooks

# %%
ds_suffix = "-uk"
max_seq_len = 20
min_seq_len = 5

# %% [markdown]
# # Process

# %%
df = pd.read_csv("data/data.csv", parse_dates=["Date"], date_format="mixed")
df.columns = [x.lower() for x in df.columns]
df["age"] = -1  # no age in this data
df

# %% [markdown]
# ##### Sort by acct

# %%
df = df.sort_values(by=["account_id", "date"])

# %% [markdown]
# ##### Date

# %%
from datetime import datetime

df["datetime"] = df["date"]

iso = df["datetime"].dt.isocalendar()

df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["dow"] = df["datetime"].dt.dayofweek
df["year"] = df["datetime"].dt.year
df

# %%
import calendar

# dtme - days till month end
df["dtme"] = df.datetime.apply(
    lambda dt: calendar.monthrange(dt.year, dt.month)[1] - dt.day
)

# %% [markdown]
# ### Tcode

# %%
from field_config import cat_code_fields, TCODE_SEP


# create tcode by concating fields in "cat_code_fields"
def set_tcode(df, cat_code_fields):
    tcode = df[cat_code_fields[0]].astype(str)
    for ccf in cat_code_fields[1:]:
        tcode += TCODE_SEP + df[ccf].astype(str)

    df["tcode"] = tcode


set_tcode(df, cat_code_fields)

# %% [markdown]
# ##### Time delta

# %%
df["td"] = df[["account_id", "datetime"]].groupby("account_id").diff()
df["td"] = df["td"].apply(lambda x: x.days).fillna(0.0)
df

# %% [markdown]
# # Write

# %%
folders = [
    "generated_data",
    "generated_data/parts",
    "stored_data",
    "generation_results",
    "data",
    "my_lib",
]


for f in folders:
    if not os.path.exists(f):
        os.mkdir(f)

# %%
df.to_csv(f"stored_data/final_df-{ds_suffix}.csv", index=False)

# %%

# %%
