import pandas as pd
import csv
from datasets import Dataset
import logging
from sklearn.model_selection import train_test_split

def data_preprocessing(file_name, train_ratio):
    if file_name.lower() == 'mentalmanip':
        path = './datasets/MentalManip/mentalmanip_con.csv'
    elif file_name.lower() == 'detex':
        path = './datasets/DetexD/dataset.csv'
    with open(path, 'r', newline='', encoding='utf-8') as infile:
        content = csv.reader(infile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        data = list()
        columns = None
        for idx, row in enumerate(content):
            if idx == 0:
                columns = row
            else:
                data.append(row)
    dataframe = pd.DataFrame(data, columns=columns)

    # drop the ID, Technique, Vulnerability columns from the main dataframe
    if 'ID' in dataframe.columns:
        dataframe = dataframe.drop(['ID'], axis=1)
    if 'Technique' in dataframe.columns:
        dataframe = dataframe.drop(['Technique'], axis=1)
    if 'Vulnerability' in dataframe.columns:
        dataframe = dataframe.drop(['Vulnerability'], axis=1)
    
    # Rename the second column to 'labels'
    second_col_name = dataframe.columns[1]
    dataframe.rename(columns={second_col_name: 'label'}, inplace=True)
    train_df, test_df = train_test_split(dataframe, train_size=train_ratio, random_state=42)

    logging.info(f"----- Dataset Information -----")
    logging.info(f"Total size = {len(dataframe)}, True Label/ False Label ratio = {len(dataframe[dataframe['label'] == '1'])/len(dataframe[dataframe['label'] == '0']):.3f}")
    logging.info(f"Train size = {len(train_df)}, True Label/ False Label ratio = {len(train_df[train_df['label'] == '1'])/len(train_df[train_df['label'] == '0']):.3f}")
    logging.info(f"Test size = {len(test_df)}, True Label/ False Label ratio = {len(test_df[test_df['label'] == '1'])/len(test_df[test_df['label']  == '0']):.3f}")
    logging.info("")

    train = Dataset.from_pandas(train_df)
    test = Dataset.from_pandas(test_df)
    
    return train, test