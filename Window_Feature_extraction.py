from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from feature_extraction import TemporalFeatures
from sklearn.preprocessing import LabelEncoder



def SplitRowsToLists(data,cols):
    result = []
    for col in cols:
        result.append(data[col])
    return result



train_data_path = 'data/train.csv'
test_data_path = 'data/test.csv'
df_train = pd.read_csv(train_data_path)

df_test = pd.read_csv(test_data_path)
main_columns = ['X0', 'Y0', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5',
       'X6', 'Y6', 'X7', 'Y7', 'X8', 'Y8', 'X9', 'Y9', 'X10', 'Y10', 'X11',
       'Y11', 'X12', 'Y12', 'X13', 'Y13', 'X14', 'Y14', 'X15', 'Y15', 'X16',
       'Y16', 'X17', 'Y17', 'X18', 'Y18', 'X19', 'Y19', 'X20', 'Y20', 'X21',
       'Y21', 'X22', 'Y22', 'X23', 'Y23', 'X24', 'Y24', 'X25', 'Y25', 'X26',
       'Y26', 'X27', 'Y27', 'X28', 'Y28', 'X29', 'Y29', 'X30', 'Y30', 'X31',
       'Y31', 'X32', 'Y32','videoID','label']


featureless_df_train = df_train[main_columns]
featureless_df_test = df_test[main_columns]


max_id = max(featureless_df_train['videoID'])

all_features = []

# print (main_columns[:-2])
Y_test = []

for i in range(0,max_id):
    list = featureless_df_train.loc[featureless_df_train['videoID'] == i]
    Cols = SplitRowsToLists(list,main_columns[:-2])
    video_features = []
    label = list['label'].iloc[0]

    for col in Cols:
        featureExtractor = TemporalFeatures(col)
        video_features= video_features +( featureExtractor.extract_features())
    all_features.append(video_features)
    Y_test.append(label)




max_id = max(featureless_df_test['videoID'])

all_test_features = []

# print (main_columns[:-2])
Y_test_test = []

for i in range(0,max_id):
    list = featureless_df_test.loc[featureless_df_test['videoID'] == i]
    Cols = SplitRowsToLists(list,main_columns[:-2])
    video_features = []
    label = list['label'].iloc[0]

    for col in Cols:
        featureExtractor = TemporalFeatures(col)
        video_features= video_features +( featureExtractor.extract_features())
    all_test_features.append(video_features)
    Y_test_test.append(label)




labelencoder = LabelEncoder()
Y_test = labelencoder.fit_transform(Y_test)

Y_test_test = labelencoder.fit_transform(Y_test_test)

for i in range(1,15):
    print(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(all_features, Y_test)
    print(knn.score(all_test_features,Y_test_test))






