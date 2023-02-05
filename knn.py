from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Loading data
train_data_path = 'data/train.csv'
test_data_path = 'data/test.csv'

# train_columns = [ 'X_min', 'Y_min','X_max', 'Y_max', 'X_mean',
#        'Y_mean', 'X_std', 'Y_std', 'X_energy', 'Y_energy', 'X_rms', 'Y_rms',
#        'X_variance', 'Y_variance', 'X_skewness', 'Y_skewness', 'X_kurtosis',
#        'Y_kurtosis', 'X_median', 'Y_median', 'X_mode', 'Y_mode', 'X_range',
#        'Y_range']
train_columns = ['X0', 'Y0', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5',
       'X6', 'Y6', 'X7', 'Y7', 'X8', 'Y8', 'X9', 'Y9', 'X10', 'Y10', 'X11',
       'Y11', 'X12', 'Y12', 'X13', 'Y13', 'X14', 'Y14', 'X15', 'Y15', 'X16',
       'Y16', 'X17', 'Y17', 'X18', 'Y18', 'X19', 'Y19', 'X20', 'Y20', 'X21',
       'Y21', 'X22', 'Y22', 'X23', 'Y23', 'X24', 'Y24', 'X25', 'Y25', 'X26',
       'Y26', 'X27', 'Y27', 'X28', 'Y28', 'X29', 'Y29', 'X30', 'Y30', 'X31',
       'Y31', 'X32', 'Y32','videoId','label']
test_column = ['label']


df_train = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)
labelencoder = LabelEncoder()



X_train = df_train[train_columns]
Y_train = labelencoder.fit_transform(df_train[test_column])
X_test = df_test[train_columns]
Y_test = labelencoder.fit_transform(df_test[test_column])

print (Y_test)

#
# # Split into training and test set
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)
#

for i in range(1,9):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)

    print(knn.score(X_test, Y_test))

