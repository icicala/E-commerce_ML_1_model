import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import Counter


def categorrical_feature_encoding(data_fraud):
    categorical_columns = ['source', 'browser', 'sex']
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    data_encoded = pd.DataFrame(onehot_encoder.fit_transform(data_fraud[categorical_columns]))
    data_fraud = pd.concat([data_fraud, data_encoded], axis=1)
    data_fraud.drop(categorical_columns, axis=1, inplace=True)
    dependent_variable = 'class'
    data_fraud = data_fraud[[col for col in data_fraud.columns if col != dependent_variable] + [dependent_variable]]
    return data_fraud
def load_data(url):
    data_fraud = pd.read_csv(url, nrows=5000)
    return data_fraud.copy()
if __name__ == '__main__':
    data_url = 'C:\\Users\\icicala\\Desktop\\Thesis\\Thesis\Data\\EFraud_data.csv'
    data = load_data(data_url)
    encoded_data = categorrical_feature_encoding(data)
    #nan_check = encoded_data.isnull().sum()
    #count_non_fraud = data[data['class'] == 1].shape[0]

    #data split
    X = encoded_data.drop(['class'], axis=1)
    y = encoded_data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)

    #fix the int column name error
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Oversampling on imbalanced class column 0:3613, 1:387
    #print("Class distribution", Counter(y_train))
    """ 
    cluster = Counter(y_train)[0]
    majority_class_ind = y_train[y_train== 0].index

    X_majority = X_train.loc[majority_class_ind]
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(X_majority)
    X_majority['cluster'] = kmeans.predict(X_majority)

    # extract medoids
    medoids = X_majority.groupby('cluster').median()

    monority_ind = y_train[y_train == 1].index

    X_minority = X_train.loc(monority_ind)

    X_train_balanced = pd.concat([X_minority, medoids], axis=0)
    y_train_balanced = y_train.loc(X_train_balanced.index)

    print(y_train_balanced)
    """

    # oversampling = RandomOverSampler(random_state=3)
    # X_train_resampl, y_train_resampl = oversampling.fit_resample(X_train, y_train)

    # SMOTE for imbalanced class columns - very poor score investigating the problem
    smote = SMOTE(random_state=7)
    X_train_resampl, y_train_resampl = smote.fit_resample(X_train, y_train)


    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train_resampl)
    X_test = sc.transform(X_test)



    """
    [[108 771]
 [  4 117]]
Accuracy Score 0.225
Precision: 0.13175675675675674
Recall: 0.9669421487603306
f1 Score: 0.23191278493557976
    """

    # training
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=7)
    classifier.fit(X_train, y_train_resampl)

    # prediction
    y_pred = classifier.predict(X_test)

    # evaluation the model
    confmatrix = confusion_matrix(y_test, y_pred)
    accuracy_score = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)

    print(confmatrix)
    print('Accuracy Score', accuracy_score)
    print('Precision:', precision)
    print('Recall:', rec_score)
    print('f1 Score:', f_score)


