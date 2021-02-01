from datetime import datetime
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier

from sklearn import metrics
from sklearn import tree
from sklearn.tree import export_graphviz

from subprocess import call
from IPython.display import Image

df_p = pd.read_csv("test_dataset.csv")

# print(df_p["Amount"])
# iterating the columns
# for col in df_p.columns:
#     print(col)

df_p["Amount"] = (random.randint(1, 10)) * df_p["Amount"] * 0.085
df_p["Remaining"] = (random.randint(1, 10)) * df_p["Remaining"] * 0.085

# print("\n")
# print(df_p["Amount"])

print("\n")
print(df_p["Date"])

df_p["Date"] = [datetime.strptime(x, '%m/%d/%Y') for x in df_p["Date"]]

df_p_diff = []

for i in range(0, len(df_p["Date"])):
    if i == 0:
        df_p_diff.append(df_p["Date"].iloc[i] - df_p["Date"].iloc[i])
    else:
        df_p_diff.append(df_p["Date"].iloc[i] - df_p["Date"].iloc[i - 1])

df_p_diff = pd.DataFrame(df_p_diff, columns=['Days'])
df_p_diff["Days"] = pd.to_numeric(df_p_diff['Days'].dt.days, downcast='integer')

# print("\n")
# print(df_p["Date"])

print("\n")
print("Calculating differences between dates of purchases")
print("No.  Day Difference")
print(df_p_diff["Days"])

# print("\n")
# print("Printing Dataset Columns")
# for col in df_p_diff.columns:
#     print(col)

df_p["Days"] = df_p_diff["Days"]

# print("\n")
# for col in df_p.columns:
#     print(col)

# x = df_p[
#      ["Transaction Number", "Amount", "Remaining", "Receiver/Sender",
#       "Sender/Receiver Name",
#       "Category_1", "Days"]]

x = df_p[
    ["Transaction Number", "Amount", "Remaining", "Category_2", "Receiver_ID", "Days"]]

y = df_p["Extra Purchase"]

print("\n")
print(x.dtypes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

clf = RandomForestClassifier(n_estimators=100)
# clf = DecisionTreeClassifier()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

feature_imp = clf.feature_importances_

print("\n")
for i, v in enumerate(feature_imp):
    print('Feature: %0d, Score: %.5f' % (i, v))

# Extract single tree
estimator = clf.estimators_[1]

model_feature_names = ["Transaction Number", "Amount", "Remaining", "Category_2", "Receiver_ID", "Days"]
model_target_names = ["Yes", "No"]

# Export as dot file
export_graphviz(estimator, out_file='test.dot',
                feature_names=model_feature_names,
                class_names=model_target_names,
                rounded=True, proportion=False,
                precision=2, filled=True)

# Convert to png using system command (requires Graphviz)
# call(['dot', '-Tpng', 'test.dot', '-o', 'test.png', '-Gdpi=600'])

# Display in jupyter notebook
# Image(filename='test.png')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6), dpi=800)
tree.plot_tree(estimator,
               feature_names=model_feature_names,
               class_names=model_target_names,
               filled=True)

fig.savefig('test.png')
