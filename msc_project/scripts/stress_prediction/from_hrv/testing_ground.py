from sklearn.metrics import f1_score
import numpy as np

y_true = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
y_pred = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
macro_f1_score = f1_score(y_true, y_pred, average="macro")
print(f"macro_f1_score: {macro_f1_score}")

positive_f1_score = f1_score(y_true, y_pred, pos_label=1)
negative_f1_score = f1_score(y_true, y_pred, pos_label=0)
mean = np.mean((positive_f1_score, negative_f1_score))
print(f"mean: {mean}")
