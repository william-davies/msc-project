import pandas as pd


def get_sample_rate(sheet_name: str):
    all_sample_rates = pd.read_pickle(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/sample_rates.pkl"
    )
    sheet_sample_rates = all_sample_rates[sheet_name]
    assert (sheet_sample_rates.values[0] == sheet_sample_rates.values).all()
    return sheet_sample_rates[0]


def get_sheet_data(all_data: pd.DataFrame, sheet_name: str):
    sheet_data = all_data.xs(key=sheet_name, axis=1, level="sheet")


if __name__ == "__main__":
    all_data = pd.read_pickle(
        "/Users/williamdavies/OneDrive - University College London/Documents/MSc Machine Learning/MSc Project/My project/msc_project/data/Stress Dataset/dataframes/raw_data.pkl"
    )

    sheet_name = "Inf"
    sheet_data = get_sheet_data(all_data=all_data, sheet_name=sheet_name)
