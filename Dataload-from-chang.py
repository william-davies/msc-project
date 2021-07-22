# %%
# load packages
import os
import time
import numpy as np
import pandas as pd


# %%
# Define load data
def load_data(start, end):
    df = None
    df_res = None
    df_edar = None
    df_edal = None

    for i in range(start, end):
        link = "F:\\UCL\\DT\\stress_data\\BVP_RES_DATA" + "\\" + str(i) + ".xlsx"
        locals()["data" + str(i)] = pd.read_excel(link)
        # set labels
        data = locals()["data" + str(i)]
        locals()["re_1_" + str(i)] = pd.DataFrame(data.iloc[1:, 3]).T
        locals()["re_1_" + str(i)]["Label"] = "relax"
        locals()["math_e_" + str(i)] = pd.DataFrame(data.iloc[1:, 6]).T
        locals()["math_e_" + str(i)]["Label"] = "math_easy"
        locals()["re_3_" + str(i)] = pd.DataFrame(data.iloc[1:, 9]).T
        locals()["re_3_" + str(i)]["Label"] = "relax"
        locals()["math_h_" + str(i)] = pd.DataFrame(data.iloc[1:, 12]).T
        locals()["math_h_" + str(i)]["Label"] = "math_hard"
        locals()["re_5_" + str(i)] = pd.DataFrame(data.iloc[1:, 15]).T
        locals()["re_5_" + str(i)]["Label"] = "relax"
        # re_math_e1.index=['']
        df = pd.concat(
            (
                df,
                locals()["re_1_" + str(i)],
                locals()["math_e_" + str(i)],
                locals()["re_3_" + str(i)],
                locals()["math_h_" + str(i)],
                locals()["re_5_" + str(i)],
            ),
            axis=0,
            ignore_index=True,
        )
        # Respiration
        locals()["respi_1_" + str(i)] = pd.DataFrame(data.iloc[1:, 4]).T
        locals()["respi_1_" + str(i)]["Label"] = "relax"
        locals()["respi_me_" + str(i)] = pd.DataFrame(data.iloc[1:, 7]).T
        locals()["respi_me_" + str(i)]["Label"] = "math_easy"
        locals()["respi_3_" + str(i)] = pd.DataFrame(data.iloc[1:, 10]).T
        locals()["respi_3_" + str(i)]["Label"] = "relax"
        locals()["respi_mh_" + str(i)] = pd.DataFrame(data.iloc[1:, 13]).T
        locals()["respi_mh_" + str(i)]["Label"] = "math_hard"
        locals()["respi_5_" + str(i)] = pd.DataFrame(data.iloc[1:, 16]).T
        locals()["respi_5_" + str(i)]["Label"] = "relax"
        df_res = pd.concat(
            (
                df_res,
                locals()["respi_1_" + str(i)],
                locals()["respi_me_" + str(i)],
                locals()["respi_3_" + str(i)],
                locals()["respi_mh_" + str(i)],
                locals()["respi_5_" + str(i)],
            ),
            axis=0,
            ignore_index=True,
        )

        # Right EDA
        locals()["EDA_R_1_" + str(i)] = pd.DataFrame(data.iloc[1:, 19]).T
        locals()["EDA_R_1_" + str(i)]["Label"] = "relax"
        locals()["EDA_R_me_" + str(i)] = pd.DataFrame(data.iloc[1:, 21]).T
        locals()["EDA_R_me_" + str(i)]["Label"] = "math_easy"
        locals()["EDA_R_3_" + str(i)] = pd.DataFrame(data.iloc[1:, 23]).T
        locals()["EDA_R_3_" + str(i)]["Label"] = "relax"
        locals()["EDA_R_mh_" + str(i)] = pd.DataFrame(data.iloc[1:, 25]).T
        locals()["EDA_R_mh_" + str(i)]["Label"] = "math_hard"
        locals()["EDA_R_5_" + str(i)] = pd.DataFrame(data.iloc[1:, 27]).T
        locals()["EDA_R_5_" + str(i)]["Label"] = "relax"
        df_edar = pd.concat(
            (
                df_edar,
                locals()["EDA_R_1_" + str(i)],
                locals()["EDA_R_me_" + str(i)],
                locals()["EDA_R_3_" + str(i)],
                locals()["EDA_R_mh_" + str(i)],
                locals()["EDA_R_5_" + str(i)],
            ),
            axis=0,
            ignore_index=True,
        )
        # Left EDA
        locals()["EDA_L_1_" + str(i)] = pd.DataFrame(data.iloc[1:, 29]).T
        locals()["EDA_L_1_" + str(i)]["Label"] = "relax"
        locals()["EDA_L_me_" + str(i)] = pd.DataFrame(data.iloc[1:, 31]).T
        locals()["EDA_L_me_" + str(i)]["Label"] = "math_easy"
        locals()["EDA_L_3_" + str(i)] = pd.DataFrame(data.iloc[1:, 33]).T
        locals()["EDA_L_3_" + str(i)]["Label"] = "relax"
        locals()["EDA_L_mh_" + str(i)] = pd.DataFrame(data.iloc[1:, 35]).T
        locals()["EDA_L_mh_" + str(i)]["Label"] = "math_hard"
        locals()["EDA_L_5_" + str(i)] = pd.DataFrame(data.iloc[1:, 37]).T
        locals()["EDA_L_5_" + str(i)]["Label"] = "relax"
        df_edal = pd.concat(
            (
                df_edal,
                locals()["EDA_L_1_" + str(i)],
                locals()["EDA_L_me_" + str(i)],
                locals()["EDA_L_3_" + str(i)],
                locals()["EDA_L_mh_" + str(i)],
                locals()["EDA_L_5_" + str(i)],
            ),
            axis=0,
            ignore_index=True,
        )

    return df, df_res, df_edar, df_edal


# %%
# Load data
start = time.clock()

df1_BVP, df1_res, df1_edar, df1_edal = load_data(1, 7)
df2_BVP, df2_res, df2_edar, df2_edal = load_data(8, 11)
df3_BVP, df3_res, df3_edar, df3_edal = load_data(16, 24)

df_BVP = pd.concat((df1_BVP, df2_BVP, df3_BVP), axis=0, ignore_index=True)
df_Res = pd.concat((df1_res, df2_res, df3_res), axis=0, ignore_index=True)
df_EDAR = pd.concat((df1_edar, df2_edar, df3_edar), axis=0, ignore_index=True)
df_EDAL = pd.concat((df1_edal, df2_edal, df3_edal), axis=0, ignore_index=True)

elapsed = time.clock() - start
print("Time used:", elapsed)

# %%
# df_RES_data =df_Res.iloc[:,0:71609]
# df_RES_label = df_Res.iloc[:,-1]
# df_RES = pd.concat((df_RES_data,df_RES_label),axis=1)
# df_RES.to_csv("F:\\UCL\\DT\\res_concat.csv")

# %%
df_BVP.head()
df_Res.head()
df_EDAR.head()
df_EDAL.head()

# %%
# find the minimum length of BVP and respiration
start = time.clock()
zero_list = []
for indexs in df_BVP.index:
    for i in range(len(df_BVP.loc[indexs].values)):
        if df_BVP.loc[indexs].values[i] == 0:
            # print(indexs,i)
            zero_list.append(i)
            break
min_length1 = np.min(zero_list)
print(min_length1)

elapsed = time.clock() - start
print("Time used:", elapsed)

# %%
# find the minimum length of edar and edal
start = time.clock()
zero_list = []

for indexs in df_EDAR.index:
    for i in range(len(df_EDAR.loc[indexs].values)):
        if df_EDAR.loc[indexs].values[i] == 0:
            # print(indexs,i)
            zero_list.append(i)
            break
min_length2 = np.min(zero_list)
print(min_length2)

elapsed = time.clock() - start
print("Time used:", elapsed)

# %%
# train_data=pd.read_csv("F:\\UCL\\DT\\data.csv",sep=",")
# label_data=pd.read_csv("F:\\UCL\\DT\\test.csv",sep=",")

# %%
# process to the same length
df_BVP_data = df_BVP.iloc[:, 0:71609]
df_BVP_label = df_BVP.iloc[:, -1]
df_BVP = pd.concat((df_BVP_data, df_BVP_label), axis=1)

df_RES_data = df_Res.iloc[:, 0:71609]
df_RES_label = df_Res.iloc[:, -1]
df_RES = pd.concat((df_RES_data, df_RES_label), axis=1)

df_EDAR_data = df_EDAR.iloc[:, 0:1137]
df_EDAR_label = df_EDAR.iloc[:, -1]
df_EDAR = pd.concat((df_EDAR_data, df_EDAR_label), axis=1)

df_EDAL_data = df_EDAL.iloc[:, 0:1137]
df_EDAL_label = df_EDAL.iloc[:, -1]
df_EDAL = pd.concat((df_EDAL_data, df_EDAL_label), axis=1)

df_BVP.head()
df_RES.head()
df_EDAR.head()
df_EDAL.head()
