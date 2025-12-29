import pandas as pd


def read_csv(path):
    df = pd.read_csv(path)
    return df


def column_dict(columns, columns_list):
    categorical_list, numerical_list, binary_list = columns_list
    categ = dict()
    for column in columns:
        if column in categorical_list:
            categ[column] = "categorical"
        elif column in numerical_list:
            categ[column] = "numerical"
        elif column in binary_list:
            categ[column] = "binary"
    return categ
