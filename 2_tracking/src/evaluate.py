import pandas as pd
import json
import copy
import numpy as np


def from_json_to_df(json_file_path):
    """
    Convert the classification json file into a simple and usable dataframe.
    We remove the key "species count" because it is useless here.
    We return the json file as a dataframe.
    """
    with open(json_file_path) as f:
        d = json.load(f)
        del d["species_count"]
    return pd.DataFrame.from_dict(d)


def merge_ground_truth_classif(df_ground_truth, df_classification):
    """
    Merge the dataframe containing the ground truth and the classification results based on the source image cropped (from 1 to 1000)
    """
    df_merged = df_classification.merge(
        df_ground_truth[["source_image_cropped", "taxon_rank", "taxon_name"]],
        how="left",
        on="source_image_cropped",
    )
    df_merged.rename(
        columns={
            "taxon_rank": "taxon_rank_ground_truth",
            "taxon_name": "taxon_name_ground_truth",
        },
        inplace=True,
    )
    return df_merged


def smoothing_max_value(df):
    """ """
    max_confidence_score = df["confidence"].max()
    max_confidence_index = df["confidence"].idxmax()
    best_label = df.iloc[max_confidence_index, df.columns.get_loc("label")]
    return max_confidence_score, best_label


def smoothing_most_frequent(df):
    # display(df)
    if df.shape[0] > 1:
        best_label = df.label.mode()[0]
        max_confidence_score = df[df["label"] == best_label]["confidence"].max()
    else:
        max_confidence_score, best_label = smoothing_max_value(df)
    return max_confidence_score, best_label


def smoothing_avg_max(df):
    avgs = df.groupby(["label"])["confidence"].mean()
    max_confidence_score = avgs.max()
    best_label = avgs.idxmax()
    return max_confidence_score, best_label


def smooth_results(df_merged):
    # test = df_evaluation.loc['000001.jpg']['track_info']
    df = df_merged[df_merged["taxon_rank_ground_truth"] == "species"]
    smoothing_labels_max = []
    scores_max = []
    smoothing_labels_most_frequent = []
    scores_most_frequent = []
    smoothing_labels_avg_max = []
    scores_avg_max = []
    for row in df["track_info"]:
        df_row = pd.DataFrame(
            row, columns=["img", "x1", "x2", "y1", "y2", "label", "confidence"]
        )
        # Max score
        score_max, label_max = smoothing_max_value(df_row)
        scores_max.append(score_max)
        smoothing_labels_max.append(label_max)
        # Most frequent
        score_most_frequent, label_most_frequent = smoothing_most_frequent(df_row)
        scores_most_frequent.append(score_most_frequent)
        smoothing_labels_most_frequent.append(label_most_frequent)
        # Max average
        score_avg_max, label_avg_max = smoothing_avg_max(df_row)
        scores_avg_max.append(score_avg_max)
        smoothing_labels_avg_max.append(label_avg_max)
    # Assign the label and confidence score after smoothing
    df_with_smoothing = df.assign(
        smoothing_label_max=smoothing_labels_max,
        score_max=scores_max,
        smoothing_label_most_frequent=smoothing_labels_most_frequent,
        score_most_frequent=scores_most_frequent,
        smoothing_label_avg_max=smoothing_labels_avg_max,
        score_avg_max=scores_avg_max,
    )
    return df_with_smoothing


def count_for_eval(df_with_smoothing):
    # I need to compare taxon_name_ground_truth and smoothing_label_max and smoothing_label_most_frequent
    evaluation_max = []
    evaluation_most_frequent = []
    evaluation_avg_max = []
    for index, row in df_evaluation.iterrows():
        if row.taxon_name_ground_truth == row.smoothing_label_max:
            evaluation_max.append(1)
        else:
            evaluation_max.append(0)
        if row.taxon_name_ground_truth == row.smoothing_label_most_frequent:
            evaluation_most_frequent.append(1)
        else:
            evaluation_most_frequent.append(0)
        if row.taxon_name_ground_truth == row.smoothing_label_avg_max:
            evaluation_avg_max.append(1)
        else:
            evaluation_avg_max.append(0)
    df_with_evaluation = df_with_smoothing.assign(
        evaluation_max=evaluation_max,
        evaluation_most_frequent=evaluation_most_frequent,
        evaluation_avg_max=evaluation_avg_max,
    )
    return df_with_evaluation


def main():
    # read csv and json file
    csv_file_path = "../../0_database/tracking/inat_validation_data_with_sources_and_scores-20230204.csv"  # contains the ground truth result for each cropped image
    json_file_path = "../../0_database/tracking/set2_maxim-kent.json"  # contains the classification result for each cropped image
    # create a dataframe from those files
    df_ground_truth = pd.read_csv(csv_file_path)
    df_classification = from_json_to_df(json_file_path)
    # merge the information
    df_merged = merge_ground_truth_classif(df_ground_truth, df_classification)
    # smooth the results
    df_with_smoothing = smooth_results(df_merged)
    # evaluate the results
    df_evaluation = count_for_eval(df_with_smoothing)
    # save to csv
    df_evaluation.to_csv("evaluation_species.csv")

    # print results
    print(
        f"Max: Count:{df_evaluation.evaluation_max.sum()}, Accuracy: {round(df_evaluation.evaluation_max.sum()/len(df_evaluation),3)}"
    )
    print(
        f"Most frequent: Count:{df_evaluation.evaluation_most_frequent.sum()}, Accuracy: {round(df_evaluation.evaluation_most_frequent.sum()/len(df_evaluation),3)}"
    )
    print(
        f"Max average: Count: {df_evaluation.evaluation_avg_max.sum()}, Accuracy: {round(df_evaluation.evaluation_avg_max.sum()/len(df_evaluation),3)}"
    )
    print("Number of species: ", len(df_evaluation))
