from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from feature_extraction import TemporalFeatures
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import SVC


from PoseExtraction import videoToDf


def SplitRowsToLists(data,cols):
    result = []
    for col in cols:
        result.append(data[col])
    return result
def Split_Df_To_X_Y(df,main_columns):
    
    max_id = max(df['videoID'])

    x = []
    y = []
    for i in range(0,max_id+1):
        list = df.loc[df['videoID'] == i]
        Cols = SplitRowsToLists(list,main_columns[:-2])
        video_features = []
        label = list['label'].iloc[0]
        for col in Cols:
            featureExtractor = TemporalFeatures(col)
            video_features= video_features +( featureExtractor.extract_features())
        x.append(video_features)
        y.append(label)
    return x,y

def VideoPath_toFeatures(test_video_path,main_columns):
    columns = ['X0', 'Y0', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'X5', 'Y5',
       'X6', 'Y6', 'X7', 'Y7', 'X8', 'Y8', 'X9', 'Y9', 'X10', 'Y10', 'X11',
       'Y11', 'X12', 'Y12', 'X13', 'Y13', 'X14', 'Y14', 'X15', 'Y15', 'X16',
       'Y16', 'X17', 'Y17', 'X18', 'Y18', 'X19', 'Y19', 'X20', 'Y20', 'X21',
       'Y21', 'X22', 'Y22', 'X23', 'Y23', 'X24', 'Y24', 'X25', 'Y25', 'X26',
       'Y26', 'X27', 'Y27', 'X28', 'Y28', 'X29', 'Y29', 'X30', 'Y30', 'X31',
       'Y31', 'X32', 'Y32']

    df = pd.DataFrame(columns = columns)

    df = videoToDf(df = df, video = test_video_path)


    Cols = SplitRowsToLists(df,columns)

    video_features = []

    for col in Cols:
         if col.name == main_columns[i]:
            featureExtractor = TemporalFeatures(col)
            video_features= video_features +( featureExtractor.extract_features())

    return video_features










def visualize(Y_test,Y_pred,class_names):
    actual = Y_test
    predicted = Y_pred
    cm = metrics.confusion_matrix(actual, predicted)

    ## Get Class Labels

    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize = 10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize = 10)
    plt.yticks(rotation=0)

    plt.title('Confusion matrix', fontsize=20)

    plt.savefig('ConMat24.png')
    plt.show()




def main():
    
    train_data_path = 'data/train.csv'
    test_data_path = 'data/test.csv'
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)

    class_names = np.unique(  df_train['label'])

    main_columns = [  'X11',
       'Y11', 'X12', 'Y12', 'X13', 'Y13', 'X14', 'Y14', 'X15', 'Y15', 'X16',
       'Y16', 'X17', 'Y17', 'X18', 'Y18', 'X19', 'Y19', 'X20', 'Y20', 'X21',
       'Y21', 'X22', 'Y22', 'X23', 'Y23', 'X24', 'Y24', 'X25', 'Y25', 'X26',
       'Y26', 'X27', 'Y27', 'X28', 'Y28', 'X29', 'Y29', 'X30', 'Y30', 'X31',
       'Y31', 'X32', 'Y32','videoID','label']
 




    featureless_df_train = df_train[main_columns]
    featureless_df_test = df_test[main_columns]

    X_train , Y_train = Split_Df_To_X_Y(featureless_df_train,main_columns)

    X_test , Y_test = Split_Df_To_X_Y (featureless_df_test,main_columns)



 




    labelencoder = LabelEncoder()
    Y_train = labelencoder.fit_transform(Y_train)
    Y_test = labelencoder.fit_transform(Y_test)




    classifier = SVC(kernel='linear',probability=True,C=10)
    classifier.fit(X_train,Y_train)


    Y_pred = classifier.predict(X_test)


    print(classifier.predict_proba(X_test))



    Y_pred = labelencoder.inverse_transform(Y_pred)
    Y_test = labelencoder.inverse_transform(Y_test)

    ct =0


    for i,result in enumerate(Y_pred):
        if Y_pred[i] == Y_test[i]:
         ct+=1
        else:
            print("truth: "  +  Y_test[i])
            print("pred: "  + Y_pred[i])


    print ("total :" + str(len(Y_pred)))
    print ("correct :" + str(ct))
    visualize(Y_test= Y_test, Y_pred=Y_pred, class_names=class_names)




main()