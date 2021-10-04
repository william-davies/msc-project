import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

num_examples = 16
num_features = 2
group_size = 2

groups = np.arange(num_examples)
groups = np.repeat(groups, group_size)

groups = np.random.choice(a=8, size=num_examples, replace=True)

X = np.arange(num_examples * num_features).reshape((num_examples, num_features))
y = np.arange(num_examples)
logo = LeaveOneGroupOut()
logo.get_n_splits(X, y, groups)

logo.get_n_splits(groups=groups)  # 'groups' is always required

# print(X)
# print(y)
print(groups)

for train_index, test_index in logo.split(X, y, groups):
    # print("TRAIN:", train_index, "TEST:", test_index)
    print(f"TEST GROUPS: {groups[test_index]}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print(X_train, X_test, y_train, y_test)
