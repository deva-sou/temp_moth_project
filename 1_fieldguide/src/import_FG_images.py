import pandas as pd
import sys
import os
from librairies import db
from pathlib import Path
import subprocess
import numpy as np
from tqdm import tqdm

df_total_count = pd.read_pickle(
    "/home/ubuntu/deva/FielGuide_import_images/data/pickles/df_total_count.pkl"
)
list_species_name = df_total_count["gbif_species_name"].tolist()
# print(list_species_name[:2])


def query_df_list_images_FG(specie_name):
    q = (
        """
    select * 
    from leps_images join leps_taxonomy
    on leps_taxonomy.id = leps_images.category_id 
    where leps_images.category_full_name = '%s'
    ORDER BY RANDOM()
    limit 1200
    """
        % specie_name
    )
    return pd.read_sql(q, con=db.engine)


def create_tree_folder(df):
    df = df.reset_index()
    for index, row in df.iterrows():
        # superfamily	family	subfamily	tribe		genus	species
        # superfamily = row['superfamily']
        family = row["family"]
        # subfamily = row['subfamily']
        # tribe = row['tribe']
        genus = row["genus"]
        species = row["species"]
        link_image = row["image_url"]
        photo_id = row["photo_id"]

        path = f"/home/ubuntu/fieldguide-media/{family}/{genus}/{species}"
        Path(path).mkdir(parents=True, exist_ok=True)
        return path


def run_image_download(list_images_file_txt, specie_name, path):
    print(f"Images are downloading for {specie_name}")
    path_txt = Path(list_images_file_txt)
    if path_txt.is_file():
        args = ["wget", "-q", "-c", "-i", list_images_file_txt, "-P", path]
        subprocess.run(args)
    print("... downloaded")


def get_FG_image_for_specie(list_species_name):
    for specie_name in list_species_name:
        df = query_df_list_images_FG(specie_name)
        path_list_images = f'/home/ubuntu/deva/FielGuide_import_images/data/txt/{specie_name.replace(" ", "_")}.txt'
        np.savetxt(path_list_images, df.image_url, fmt="%s")
        # create_list_image_txt(df,specie_name)
        target_directory_path = create_tree_folder(df)
        run_image_download(path_list_images, specie_name, target_directory_path)


get_FG_image_for_specie(list_species_name)
