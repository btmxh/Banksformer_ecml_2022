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
# This notebooks takes a pre-processed dataframe, and encodes the data so it can be used to train Banksformer.
#
#
# The input dataframe requires following columns:
# - tcode - String, encodes transaction type
# - amount - float, transcation amount (not log)
# - account_id - int, associates transactions with account
# - age - int, clients age
# - datetime - datetime object, date of transaction
# - day, month, dow - all ints, encode day, month and day of week
# - td - int/float, time delta, encodes number of days since the last transaction
#
# The encoded data will be tensor of shape (n_samples, max_seq_len, feats_per_step).

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import time

import pickle
from datetime import date
import tensorflow as tf

# %% [markdown]
# ## Setup

# %%
ds_suffix = "-uk"
max_seq_len = 20
min_seq_len = 5

# %% [markdown]
# ### Load dataframe

# %%
df = pd.read_csv(f"stored_data/final_df-{ds_suffix}.csv")
df

# %% [markdown]
# ## Ensure correct folders exist

# %%
folders = [
    "generated_data",
    "stored_data",
    "checkpoints",
    "generation_results",
    "data",
    "my_lib",
]


for f in folders:
    if not os.path.exists(f):
        os.mkdir(f)

# %% [markdown]
# ## Encode

# %%
df["amount"] = np.abs(
    df.amount
)  # model only magnitude of amount, use categorical feature to distinguish credit/debit

# %%
from my_lib.encoding import preprocess_df, bulk_encode_time_value
from field_config import CAT_FIELDS

preprocess_df(df, CAT_FIELDS, ds_suffix)

# %%
# from my_lib.field_config import *

from field_config import (
    get_field_info,
    DATA_KEY_ORDER,
    CLOCK_DIMS,
    INP_ENCODINGS,
    TAR_ENCODINGS,
)


def count_seqs_in_df(df):
    gb_aid = df.groupby("account_id")["account_id"]

    full_seqs_per_acct = gb_aid.count() // max_seq_len

    n_full_seqs = sum(full_seqs_per_acct)
    n_part_seqs = sum(gb_aid.count() - full_seqs_per_acct * max_seq_len >= min_seq_len)

    return n_full_seqs + n_part_seqs


def seq_to_inp_tensor(seq, inp_tensor, seq_i, seq_len):
    for k in DATA_KEY_ORDER:
        depth = FIELD_DIMS_IN[k]
        st = FIELD_STARTS_IN[k]
        enc_type = INP_ENCODINGS[k]

        if enc_type == "oh":
            x = tf.one_hot(seq[k], depth).numpy()
        elif enc_type == "cl":
            max_val = CLOCK_DIMS[k]
            x = bulk_encode_time_value(seq[k], max_val)
        elif enc_type == "raw":
            x = np.expand_dims(seq[k], 1)
        else:
            raise Exception(f"Got invalid enc_type: {enc_type}")

        inp_tensor[seq_i, :seq_len, st : st + depth] = x


def seq_to_targ_tensor(seq, tar_tensor, seq_i, seq_len):
    for k in DATA_KEY_ORDER:
        depth = FIELD_DIMS_TAR[k]
        st = FIELD_STARTS_TAR[k]
        enc_type = TAR_ENCODINGS[k]

        if enc_type == "cl-i":
            max_val = CLOCK_DIMS[k]
            x = np.expand_dims(seq[k] % max_val, 1)
        elif enc_type == "raw":
            x = np.expand_dims(seq[k], 1)
        else:
            raise Exception(f"Got invalid enc_type: {enc_type}")

        tar_tensor[seq_i, :seq_len, st : st + depth] = x


(
    FIELD_DIMS_IN,
    FIELD_STARTS_IN,
    FIELD_DIMS_TAR,
    FIELD_STARTS_TAR,
    FIELD_DIMS_NET,
    FIELD_STARTS_NET,
) = get_field_info(ds_suffix)

# %%
n_seqs = count_seqs_in_df(df)
n_steps = max_seq_len
n_feat_inp = sum(FIELD_DIMS_IN.values())
n_feat_tar = sum(FIELD_DIMS_TAR.values())

inp_tensor = np.zeros((n_seqs, n_steps, n_feat_inp))
tar_tensor = np.zeros((n_seqs, n_steps, n_feat_tar))

inp_tensor.shape, tar_tensor.shape

# %%
seq_i = 0
rows_per_acct = {}
alert_every = 2000
attribute = "age_sc"


attributes = np.zeros(n_seqs)
start_time = time.time()
for acct_id, group in df.groupby("account_id"):
    rows_per_acct[acct_id] = []

    for i in range(len(group) // max_seq_len + 1):
        n_trs = len(group)
        start = i * max_seq_len
        seq_len = min(max_seq_len, n_trs - start)

        if seq_len >= min_seq_len:
            seq_to_inp_tensor(
                group.iloc[start : start + seq_len], inp_tensor, seq_i, seq_len
            )
            seq_to_targ_tensor(
                group.iloc[start : start + seq_len], tar_tensor, seq_i, seq_len
            )
            #             tar_tensor[seq_i,:seq_len,:] = seq_to_targ_tensor(group.iloc[start:start+seq_len])
            attributes[seq_i] = group["age"].iloc[0]

            rows_per_acct[acct_id].append(seq_i)
            seq_i += 1

            if seq_i % alert_every == 0:
                print(f"Finished encoding {seq_i} of {n_seqs} seqs")


# Add conditioning info (attribute) to first timestep of inp
inp_tensor = np.concatenate(
    [np.repeat(attributes[:, None, None], n_feat_inp, axis=2), inp_tensor], axis=1
)
print(f"Took {time.time() - start_time:.2f} secs")

# %%
inp_tensor.shape, tar_tensor.shape, attributes.shape

# %% [markdown]
# ## Save

# %%
np.save(f"stored_data/inp_tensor-{ds_suffix}", inp_tensor)
np.save(f"stored_data/tar_tensor-{ds_suffix}", tar_tensor)
np.save(f"stored_data/attributes-{ds_suffix}", attributes)

# %%
with open(f"stored_data/rows_per_acct-{ds_suffix}.pickle", "wb") as f:
    pickle.dump(rows_per_acct, f)
