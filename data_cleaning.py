from pyspark.sql import SparkSession
# from pyspark.sql.functions import *
import os

data_dir_path = 'data'
spark = SparkSession.builder.appName("merging and cleaning").getOrCreate()


def read_in_data():
    '''
    Iterates through the files in the data directory, and converts them to spark data structures.
    ------------------------------------------------------
    Returns a dict of dicts {{}}, where the top level keys are:
    - 'data' (files for train, test, val)
    - 'aux': auxiliary data to be joined with the train, test val data
    Each of 'data' and 'aux' are dictionaries as well, where:
    - the keys are the file names for 'aux', and 'train'/'test'/'validation' for 'data'
    - the values are the data, as a spark structure
    '''
    data_spark_aux = {}
    data_spark = {}

    train_paths = []
    for filename in os.listdir(data_dir_path):
        full_path = os.path.join(data_dir_path, filename)
        file_name = filename.split('.')[0].replace('-', '_')

        if 'train' in file_name:
            train_paths.append(full_path) # we collect the paths here since we need to stack them all
        elif 'test' in file_name:
            data_spark['test'] = spark.read.csv(full_path, header=True, inferSchema=True)
        elif 'validation' in file_name:
            data_spark['validation'] = spark.read.csv(full_path, header=True, inferSchema=True)
        else:
            data_spark_aux[file_name] = spark.read.json(full_path)

    data_spark['train'] = stack_train(train_paths)
    
    return {'data': data_spark, 'aux': data_spark_aux}



def stack_train(train_paths):
    stacked_df = spark.read.csv(train_paths[0], header=True, inferSchema=True)
    for t in train_paths[:-1]:
        df_pyspark = spark.read.csv(t, header=True, inferSchema=True)
        stacked_df = stacked_df.union(df_pyspark) 
    return stacked_df


all_data_map = read_in_data()
print('*** aux files ***')
print(all_data_map['aux'].keys())
print('*** train-test-val ***')
print(all_data_map['data'].keys())
print(all_data_map['data']['test'].columns)

# TODO next @lori: check out the aux data formats and merge the aux data with the train/val/test data

