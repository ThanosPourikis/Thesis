import datetime
import json
import os
import urllib
from os import listdir
from os.path import join
import pandas as pd
import requests
from pytz import timezone


def target_model_data(folder_name="all_files_isp1"):
    if not os.path.isdir(folder_name):
        print("create", folder_name)
        os.makedirs(folder_name)
        onlyfiles = []
        print("Folder created")
    else:
        mypath = folder_name
        print("dir existed")
        onlyfiles = [f for f in listdir(mypath) if os.path.isfile(join(mypath, f))]
    dt = datetime.datetime.now() + datetime.timedelta(days=1)
    month = dt.month
    day = dt.day
    year = dt.year
    target_data = "https://www.admie.gr/getOperationMarketFilewRange?dateStart=2020-11-01&dateEnd=%s-%s-%s&FileCategory=ISP1Requirements" % (
    year, month, day)
    retrieved_data = requests.get(target_data, stream=True)
    files = json.loads(retrieved_data.text)
    dataframe_dictionary = {}
    for file in files:
        file_desc = file["file_description"].split(" ")[0]
        # file_desc = datetime.datetime.strptime(file_desc, '%Y%m%d')
        fp = file["file_path"]
        if fp.endswith("01.xlsx"):
            print("file selected")
            filename = fp.split("/")[-1]
            fp = file["file_path"]
            if fp.split("/")[-1] in onlyfiles:
                dpp = "%s/" % folder_name + filename
                dataframe_dictionary[file_desc] = pd.read_excel(dpp)
                continue
            urllib.request.urlretrieve(fp, "%s/" % folder_name + filename)
            dpp = "%s/" % folder_name + filename
            dataframe_dictionary[file_desc] = pd.read_excel(dpp)
    cor = {3: "Res_Total", 7: "Load Total", 27: "Hydro Total"}
    exporting_df = {}
    print("files downloaded")
    all_imp_exp = dict()
    up_down_dict = dict()
    for key in dataframe_dictionary:
        idx = dataframe_dictionary[key][dataframe_dictionary[key].iloc[:, 0] == 'Reserve Requirements'].index[0]
        end_df = dataframe_dictionary[key].shape[0] - 1  # since the counting starts from 0
        idx_down = dataframe_dictionary[key][dataframe_dictionary[key].iloc[:, 0] == 'Down'].index[0]
        up_rows = list(range(idx + 1, idx_down))
        down_rows = list(range(idx_down, end_df))
        up_down_dict[key] = {"up": up_rows, "down": down_rows}
        dataframe_dictionary[key] = dataframe_dictionary[key].iloc[:, :-1]
        dataframe_dictionary[key] = dataframe_dictionary[key].iloc[:-1, :]
        kept = dataframe_dictionary[key].iloc[[3, 7, 27], 2:].copy()
        kept["index"] = kept.index
        kept["names"] = kept["index"].apply(lambda x: cor[x])
        kept.index = kept["names"]
        del kept["names"]
        del kept["index"]
        other_range = dataframe_dictionary[key].iloc[idx + 1:, :].copy()
        other_range = other_range.iloc[:, 2:]
        exporting_df[key] = pd.concat([kept, other_range])
        pairs = []
        for x in range(0, exporting_df[key].shape[1], 2):
            pairs.append(exporting_df[key].iloc[:, x:x + 2].mean(axis=1))
        exporting_df[key] = pd.concat(pairs, axis=1)
        all_imp_exp[key] = up_rows + down_rows
    to_keep = ['Res_Total', 'Load Total', 'Hydro Total', 'Date', 'sum_imports', 'sum_exports']

    localtz = timezone('CET')
    flag = False
    for key in exporting_df:
        upper = exporting_df[key].shape[1] + 1
        if upper > 25:
            kept_cols = list(set(exporting_df[key].columns) - {3})  # essentially remove 03:00 o clock.
            exporting_df[key] = exporting_df[key].iloc[:, kept_cols]
        elif upper < 25:
            flag = True
            exporting_df[key].insert(3, 2.5, 0)
            exporting_df[key].columns = list(range(exporting_df[key].shape[-1]))
        upper = 25
        change_df = exporting_df[key].T
        timestamp_date = datetime.datetime.strptime(key, '%Y%m%d')
        tmp = [timestamp_date + datetime.timedelta(hours=i) for i in range(1, upper)]
        tmp = [localtz.localize(unaware) for unaware in tmp]
        change_df["Date"] = tmp
        change_df.loc[:, "sum_imports"] = change_df.loc[:, up_down_dict[key]["up"]].sum(axis=1)
        change_df.loc[:, "sum_exports"] = change_df.loc[:, up_down_dict[key]["down"]].sum(axis=1)
        change_df = change_df.drop(all_imp_exp[key], axis=1).reset_index(drop=True)
        if flag:
            change_df.iloc[2, 3] = change_df["Date"].iloc[3]
            change_df = change_df.drop_duplicates(subset=["Date"], keep="first")
        exporting_df[key] = change_df.loc[:, to_keep]
    exporting_df = pd.concat(exporting_df.values()).sort_values(by="Date").reset_index(drop=True)
    return exporting_df
