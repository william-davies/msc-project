import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = np.load("Stress Dataset/dataset_two_min_window.npy")
data = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    data, data, test_size=0.2, random_state=42
)
