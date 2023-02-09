import json
import pandas as pd


def from_json_to_df(json_file_path):
    """
    Convert the classification json file into a simple and usable dataframe.
    We remove the key "species count" because it is useless here.
    Then to extract specific information of classification you can do df['000001.jpg']['track_info'] and then select the image
    """
    with open(json_file_path) as f:
        d = json.load(f)
        del d["species_count"]
    return pd.DataFrame.from_dict(d)


csv_file_path = "../../0_database/tracking/inat_validation_data_with_sources_and_scores-20230204.csv"
json_file_path = "../../0_database/tracking/set2_maxim-kent.json"
df_gt = pd.read_csv(csv_file_path)
df_classification = from_json_to_df(json_file_path)
df_evaluation = pd.DataFrame(
    columns=[
        "source_image_original",
        "source_image_cropped",
        "ground_truth_taxon_rank",
        "ground_truth_taxon_name",
        "prediction_made",
        "confidence_score",
        "evaluation",
    ]
)
