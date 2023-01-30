import pandas as pd
import copy
import numpy as np
import pickle

def calculate_best_confidence_max_value(df,df_specific_id):
    '''
    Calculate the best confidence score based on the strategy max value. 
    @params df: data where 1 row is an image where tracking and classification where made 
    @paramns df_specific_id: data filtered on a specific track id
    Best test: track_id == 1
    '''
    max_confidence_index = df_specific_id['confidence'].idxmax()
    new_label = df.iloc[max_confidence_index,df.columns.get_loc("subclass")]
    return max_confidence_index,new_label

def calculate_best_confidence_most_frequent(df,df_specific_id):
    '''
    Calculate the best confidence score based on the strategy most frequent
    @params df: data where 1 row is an image where tracking and classification where made 
    @paramns df_specific_id: data filtered on a specific track id
    Best test: track_id == 39
    '''
    most_frequent_species = df_specific_id.subclass.mode()
    if most_frequent_species.size != 1:
        pass # TO DO: if two most values -> get the one with best confidence
    else:
        new_label = most_frequent_species[0]
    most_frequent_specie = df_specific_id[df_specific_id['subclass'] == new_label]
    max_confidence_index = most_frequent_specie['confidence'].idxmax()
    return max_confidence_index,new_label

def calculate_best_confidence_avg_max(df,df_specific_id):
    '''
    Calculate the mean for each unique value fo subclass
    Return the specie name and the index
    '''
    avgs = df_specific_id.reset_index(level=0).groupby(['subclass','index']).agg({'confidence': ['mean']})
    #avgs.to_pickle("avgs.pkl")

    #first get label from avgmax
    df_mean = avgs.groupby(['subclass']).mean()
    new_label = df_mean.confidence.idxmax().values[0]
    
    #second get the avg of the max confident label
    new_confidence_score = df_mean.loc[new_label].values[0]
    
    #third get the max confidence index
    max_confidence_index = avgs['confidence'].idxmax()
    
    #replace the avg max for each iteration of label
    df.loc['Hemaris gracilis',2082] = 100

    print(new_label, new_confidence_score)
    print(max_confidence_index)

    return 0,new_label

def modify_label_confidence(df,df_specific_id):
    '''
    Modify the label and confidence depending on the strategies
    '''
    max_confidence_index,new_label = calculate_best_confidence_avg_max(df,df_specific_id)
    #print(f"Selected index and value: {new_label}: {max_confidence_index}")
    for index, row in df_specific_id.iterrows():
        if (index != max_confidence_index) : # & (row['subclass'] != new_label)
            df.iloc[index, df.columns.get_loc("subclass")] = new_label
            df.iloc[index, df.columns.get_loc("confidence")] = 'smoothed'
        
def smoothing_confidence_with_tracking(df_raw):
    '''
    Use the tracking insect results to smooth the classification.
    For each trackID, we created a sub-dataframe based on the specific trackID. 
    Then modify labels and confidence based on a specific strategy with #*modify_label_confidence
    #TODO: replace empty by nan
    '''
    df = df_raw.copy()
    #max_id_tracked = int(df['track_id'].max())
    max_id_tracked = 39 #To test a specific trackID
    for track_id in range(39,max_id_tracked+1):
        df_ids_raw = df[df['track_id'] == track_id]
        if df_ids_raw.shape[0] > 1:
            df_specific_id = df_ids_raw.astype({'confidence': int}) #necessary, otherwise: confidence = object instead of int
            modify_label_confidence(df,df_specific_id)
    return df

def count_species(csv):
    '''
    Count unique individual per species
    @params csv: take a csv
    '''
    df = pd.read_csv(csv)
    df_count = df.groupby(['subclass']).size().sort_values(ascending=False)
    df_count.to_csv('count.txt',header=False)
    return df_count

def main():
    df_raw = pd.read_csv("~/code/moth_project/0_database/tracking/track_localize_classify_annotation-2022_05_13.csv")
    df_modified = smoothing_confidence_with_tracking(df_raw)
    return df_modified
