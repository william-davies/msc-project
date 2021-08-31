import pandas as pd

all_data = pd.read_pickle(
    "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/raw_data.pkl"
)

all_data.xs(key="Inf", axis=1, level="sheet")
all_data.xs(key=["EmLBVP", "EmRBVP"], axis=1, level="sheet")

all_data.loc[:, pd.IndexSlice[:, ["EmLBVP", "EmRBVP"]]]
