from utils import read_csv, column_dict
import yaml

path = "data/celva_transl.csv"

df = read_csv(path)

with open("config/graph_config.yaml", "r") as f:
    graph_config = yaml.load(f, Loader=yaml.SafeLoader)

with open("config/df_config.yaml", "r") as f:
    df_config = yaml.load(f, Loader=yaml.SafeLoader)

graph_columns = graph_config["graph_columns"]
threshold = graph_config["threshold"]

df_columns = [
    df_config["categorical_list"],
    df_config["numerical_list"],
    df_config["binary_list"],
]

cat = column_dict(graph_columns, df_columns)
print(cat)
