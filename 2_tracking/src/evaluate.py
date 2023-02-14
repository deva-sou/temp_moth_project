import pandas as pd
import json
import copy
import numpy as np


def from_json_to_df(json_file_path):
    """
    Convert the classification json file into a simple and usable dataframe.
    We remove the key "species count" because it is useless here.
    Then to extract specific information of classification you can do df['000001.jpg']['track_info'] and then select the image
    """
    with open(json_file_path) as f:
        d = json.load(f)
        del d["species_count"]
    df = (
        pd.DataFrame.from_dict(d, orient="index")
        .reset_index(level=0)
        .rename(columns={"index": "source_image_cropped"})
    )
    return df


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


def smoothing_max_value(df_prediction_for_one_image):
    """
    For multiple prediction of one cropped image that represent one specie, you take the maximum confidence score.
    Return the score and the best label.
    """
    max_confidence_score = df_prediction_for_one_image["confidence"].max()
    max_confidence_index = df_prediction_for_one_image["confidence"].idxmax()
    best_label = df_prediction_for_one_image.iloc[
        max_confidence_index, df_prediction_for_one_image.columns.get_loc("label")
    ]
    return max_confidence_score, best_label


def smoothing_most_frequent(df_prediction_for_one_image):
    """
    For multiple prediction of one cropped image that represent one specie, take the max confidence score and the label of the most frequent specie.
    Return the score and the best label.
    TODO: the if is not precise, I need to detect if there is at least 1 non unique value. Then apply the smooth. If there is only unique values, do the max.
    """
    if df_prediction_for_one_image.shape[0] > 1:
        best_label = df_prediction_for_one_image.label.mode()[0]
        max_confidence_score = df_prediction_for_one_image[
            df_prediction_for_one_image["label"] == best_label
        ]["confidence"].max()
    else:  # if there is only unique species, take the max value
        max_confidence_score, best_label = smoothing_max_value(
            df_prediction_for_one_image
        )
    return max_confidence_score, best_label


def smoothing_avg_max(df_prediction_for_one_image):
    """
    For multiple prediction of one corpped image that represent one specie, take the max average confidence score and the label corresponding
    Return the score and the best label.
    """
    avgs = df_prediction_for_one_image.groupby(["label"])["confidence"].mean()
    max_confidence_score = avgs.max()
    best_label = avgs.idxmax()
    return max_confidence_score, best_label


def smooth_results(df_merged):
    """
    Smooth the result based on the three strategies, max, most frequent and avg max.
    Return a dataframe with new columns containing results for each strategies.
    """
    # Filter only the ground truth of taxon rank as species
    df = df_merged[df_merged["taxon_rank_ground_truth"] == "species"]
    # Create lists to store the result of each smoothing strategy.
    smoothing_labels_max = []
    scores_max = []
    smoothing_labels_most_frequent = []
    scores_most_frequent = []
    smoothing_labels_avg_max = []
    scores_avg_max = []
    # Iterate of each row to extract information of each cropped image as 000001.jpg
    for row in df["track_info"]:
        # Create a dataframe from the row to extract each label and confidence score available
        df_row = pd.DataFrame(
            row, columns=["img", "x1", "x2", "y1", "y2", "label", "confidence"]
        )
        ## Apply strategies
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
    # Assign the label and confidence score after smoothing on new columns
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
    """
    After smoothing the results, apply an evaluation and calculate the accuracy.
    For each strategy, if the best label is equal to the ground truth, on a new column add the value 1. If not, put 0.
    At the end sum all the 1 score and divide by the lenght of the dataframe. If 5 out of 10 new classification is correct, the accuracy is 0.5
    """
    # Create list to store the results.
    evaluation_max = []
    evaluation_most_frequent = []
    evaluation_avg_max = []
    # Iterate for each cropped image and apply the evaluation
    for index, row in df_with_smoothing.iterrows():
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
    # Add the results to a data frame
    df_evaluation = df_with_smoothing.assign(
        evaluation_max=evaluation_max,
        evaluation_most_frequent=evaluation_most_frequent,
        evaluation_avg_max=evaluation_avg_max,
    )
    # Get the stats
    number_of_image_tested = len(df_evaluation)
    max_count = df_evaluation.evaluation_max.sum()
    accuracy_max = round(max_count / number_of_image_tested, 3)

    most_frequent_count = df_evaluation.evaluation_max.sum()
    accuracy_most_frequent = round(most_frequent_count / number_of_image_tested, 3)

    max_avg_count = df_evaluation.evaluation_max.sum()
    accuracy_max_avg = round(max_avg_count / number_of_image_tested, 3)

    return (
        df_evaluation,
        max_count,
        accuracy_max,
        most_frequent_count,
        accuracy_most_frequent,
        max_avg_count,
        accuracy_max_avg,
    )


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
    (
        df_evaluation,
        max_count,
        accuracy_max,
        most_frequent_count,
        accuracy_most_frequent,
        max_avg_count,
        accuracy_max_avg,
    ) = count_for_eval(df_with_smoothing)
    # save to csv
    df_evaluation.to_csv("evaluation_species.csv")

    # print results
    print(f"Max: Count: {max_count}, Accuracy: {accuracy_max}")
    print(
        f"Most frequent: Count: {most_frequent_count}, Accuracy: {accuracy_most_frequent}"
    )
    print(f"Max average: Count: {max_avg_count}, Accuracy: {accuracy_max_avg}")
    print("Number of species: ", len(df_evaluation))


if __name__ == "__main__":
    main()
