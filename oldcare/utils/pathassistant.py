import pandas as pd
import os

path_info_csv = 'D:\CodingProject\PyCharmProject\CIS-CV\info\path_info.csv'


def get_path(path_name, type=0):
    df = pd.read_csv(path_info_csv)
    paths = {}
    for i in range(df.shape[0]):
        values = df.values
        paths[values[i, 0]] = values[i, 1]
    if type == 0:
        return os.path.join(paths['absolute_project_path'], paths[path_name])
    else:
        return os.path.join(paths['absolute_snapshot_path'], paths[path_name])