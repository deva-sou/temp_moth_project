import pandas as pd
import copy
import numpy as np
import pickle

# Strategies
def calculate_best_confidence_max_value(df, df_specific_id):
    """
    Calculate the best confidence score based on the strategy max value.
    @params df: data where 1 row is an image where tracking and classification where made
    @paramns df_specific_id: data filtered on a specific track id
    Best test: track_id == 1
    """
    max_confidence_index = df_specific_id["confidence"].idxmax()
    new_label = df.iloc[max_confidence_index, df.columns.get_loc("subclass")]
    return df_specific_id, max_confidence_index, new_label


def calculate_best_confidence_most_frequent(df, df_specific_id):
    """
    Calculate the best confidence score based on the strategy most frequent
    @params df: data where 1 row is an image where tracking and classification where made
    @paramns df_specific_id: data filtered on a specific track id
    Best test: track_id == 39
    """
    most_frequent_species = df_specific_id.subclass.mode()
    if most_frequent_species.size != 1:
        pass  # TO DO: if two most values -> get the one with best confidence
    else:
        new_label = most_frequent_species[0]
    most_frequent_specie = df_specific_id[df_specific_id["subclass"] == new_label]
    max_confidence_index = most_frequent_specie["confidence"].idxmax()
    return df_specific_id, max_confidence_index, new_label


def calculate_best_confidence_avg_max(df, df_specific_id):
    """
    Calculate the mean for each unique value fo subclass
    Return the specie name and the index
    """
    # first create the subdataframe with the indexes saved.
    avgs = (
        df_specific_id.reset_index()
        .groupby(["subclass", "index"])
        .agg({"confidence": ["mean"]})
        .reset_index()
        .set_index("index")
    )
    avgs.columns = ["subclass", "confidence"]
    avgs.index.name = None
    avgs.to_pickle("../../0_database/results/avgs_raw.pkl")

    # second get the avg of the max confident label
    df_mean = avgs.assign(mean=avgs.groupby(["subclass"]).transform("mean"))
    df_mean = df_mean.drop(columns=["confidence"])
    df_mean = df_mean.rename(columns={"mean": "confidence"})
    df_mean.to_pickle("../../0_database/results/df_mean.pkl")

    # third get the max confidence index and new label
    max_confidence_index = df_mean["confidence"].idxmax()
    new_label = df_mean.loc[max_confidence_index].values[0]

    return df_mean, max_confidence_index, new_label


# Main
def modify_label_confidence(df, df_specific_id):
    """
    Modify the label and confidence depending on the strategies.
    From the dataframe specific to a trackID, we extract the best choice by the classification algorithm for a specific choice strategy
    Then for each row of the specific dataframe we extract the index and make changes in the base dataframe.

    """
    (
        df_specific_id_modified,
        max_confidence_index,
        new_label,
    ) = calculate_best_confidence_avg_max(df, df_specific_id)
    for index, row in df_specific_id_modified.iterrows():
        if index != max_confidence_index:
            df.iloc[index, df.columns.get_loc("subclass")] = new_label
            df.iloc[index, df.columns.get_loc("confidence")] = "smoothed"


def smoothing_confidence_with_tracking(df_raw):
    """
    Use the tracking insect results to smooth the classification.
    For each trackID, we created a sub-dataframe based on the specific trackID.
    Then modify labels and confidence based on a specific strategy with "modify_label_confidence()"
    #TODO: replace empty by nan
    """
    df = df_raw.copy()
    # max_id_tracked = int(df['track_id'].max())
    max_id_tracked = 39  # To test a specific trackID
    for track_id in range(39, max_id_tracked + 1):
        df_ids_raw = df[df["track_id"] == track_id]
        if df_ids_raw.shape[0] > 1:
            df_specific_id = df_ids_raw.astype(
                {"confidence": int}
            )  # necessary, otherwise: confidence = object instead of int
            modify_label_confidence(df, df_specific_id)
    return df


def count_species(csv):
    """
    Count unique individual per species
    """
    df = pd.read_csv(csv)
    df_count = df.groupby(["subclass"]).size().sort_values(ascending=False)
    df_count.to_csv("count.txt", header=False)
    return df_count


def main():
    df_raw = pd.read_csv(
        "../../0_database/tracking/track_localize_classify_annotation-2022_05_13.csv"
    )
    df_modified = smoothing_confidence_with_tracking(df_raw)
    return df_modified
