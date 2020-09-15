# -*- coding: UTF-8 -*-
import warnings
warnings.filterwarnings('ignore')  
import pandas as pd 
import os
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns 
from time import time
from sklearn.preprocessing import scale  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score 
import xgboost as xgb  
from sklearn.svm import SVC  
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV  
from sklearn.metrics import make_scorer 
import joblib  
import numpy as np
import scipy.io as io
loc = 'F://Python//archive//' 
res_name = []  
filecsv_list = []  
def file_name(file_name):
    for root,dirs,files in os.walk(file_name):
        files.sort()
        for i,file in enumerate(files):
            if os.path.splitext(file)[1] == '.csv':
                filecsv_list.append(file)
                res_name.append('raw_data_'+str(i+1))
    print(res_name)
    print(filecsv_list)
file_name(loc)

time_list = [filecsv_list[i][0:4]  for i in range(len(filecsv_list))]
time_list
for i in range(len(res_name)):
    res_name[i] = pd.read_csv(loc+filecsv_list[i],error_bad_lines=False)
for i in range(len(res_name),0,-1): 
    if res_name[i-1].shape[0] != 380:
        key = 'res_name[' + str(i) + ']'
        res_name.pop(i-1)
        time_list.pop(i-1)
        continue
res_name[0].head()
res_name[0].tail()
res_name[0]['HomeTeam'].unique()
shape_list = [res_name[i].shape[1] for i in range(len(res_name))]
for i in range(len(res_name)):
    if res_name[i].shape[1] == max(shape_list):
# 2. Data cleaning and pretreatment
# 2.1 Pick information column
# Put selected information in a new list
columns_req = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR']
playing_statistics = []  
playing_data = {}  
for i in range(len(res_name)):
    playing_statistics.append('playing_statistics_'+str(i+1))
    playing_statistics[i] = res_name[i][columns_req]
    print(time_list[i],'playing_statistics['+str(i)+']',playing_statistics[i].shape)
# 2.2 Analyzing raw data
# 2.2.1 Count the accuracy of all home teams will win
def predictions_0(data):
    predictions = []
    for _, game in data.iterrows():

        if game['FTR']=='H':
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)

avg_acc_sum = 0
for i in range(len(playing_statistics)):
    predictions = predictions_0(playing_statistics[i])
    acc=sum(predictions)/len(playing_statistics[i])
    avg_acc_sum += acc
print("%sThe accuracy of the annual data home wins prediction is%s"%(time_list[i],acc))
print(' The average accuracy of the total %s is ：%s'%(len(playing_statistics),avg_acc_sum/len(playing_statistics)))
# 2.2.2 Count the accuracy of all away teams will win
def predictions_1(data):
    predictions = []
    for _, game in data.iterrows():

        if game['FTR']=='A':
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)

for i in range(len(playing_statistics)):
    predictions = predictions_1(playing_statistics[i])
    acc=sum(predictions)/len(playing_statistics[i])
    print("%sThe accuracy of the annual data away win prediction is%s"%(time_list[i],acc))

# 2.3 We want to know how Arsenal's performance as a home team, how to find the cumulative number of goals in all competitions in 20011-12?
def score(data):
    scores=[]
    for _,game in data.iterrows():
        if game['HomeTeam']=='Arsenal':
            scores.append(game['FTHG'])
    return np.sum(scores)
Arsenal_score=score(playing_statistics[2])
print("Arsenal as the home team in 2010, the cumulative number of goals：%s"%(Arsenal_score))
# 2.4 We want to know how their performances are when teams play as home teams.
print(playing_statistics[5].groupby('HomeTeam').sum()['FTHG'])
# 3. Feature engineering
# 3.1 Structural feature
# 3.1.1 Calculate the cumulative number of goal difference per team week
def get_goals_diff(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']

        teams[playing_stat.iloc[i].HomeTeam].append(HTGS-ATGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS-HTGS)

    GoalsDifference = pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T
    GoalsDifference[0] = 0
    for i in range(2,39):
        GoalsDifference[i] = GoalsDifference[i] + GoalsDifference[i-1]
    return GoalsDifference

def get_gss(playing_stat):
    GD = get_goals_diff(playing_stat)
    j = 0
    HTGD = []
    ATGD = []
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGD.append(GD.loc[ht][j])
        ATGD.append(GD.loc[at][j])
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat.loc[:,'HTGD'] = HTGD
    playing_stat.loc[:,'ATGD'] = ATGD
    return playing_stat

for i in range(len(playing_statistics)):
    playing_statistics[i] = get_gss(playing_statistics[i])

# 3.1.2  Statistics of the cumulative score of the home and away team to the current game week
# Convert the result of the game into a score, win three points, score one point, lose no points
def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2,39):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
    matchres_points.insert(column =0, loc = 0, value = [0*i for i in range(20)])
    return matchres_points

def get_matchres(playing_stat):
# Create a dictionary with each team's name as the key
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    # Record the results of the competition in the home team and the away team
# H: On behalf of the home win     
# A: On behalf of the away win     
# D: Representative Draw    
for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
    return pd.DataFrame(data=teams, index = [i for i in range(1,39)]).T

def get_agg_points(playing_stat):
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat.loc[:,'HTP'] = HTP
    playing_stat.loc[:,'ATP'] = ATP
    return playing_stat

for i in range(len(playing_statistics)):
    playing_statistics[i] = get_agg_points(playing_statistics[i])
playing_statistics[2].tail()
# 3.1.3 Statistics on the performance of a team in the last three games
def get_form(playing_stat,num):
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num,39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final

def add_form(playing_stat,num):
    form = get_form(playing_stat,num)
    h = ['M' for i in range(num * 10)]
    a = ['M' for i in range(num * 10)]
    j = num
    for i in range((num*10),380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        past = form.loc[ht][j]
        h.append(past[num-1])

        past = form.loc[at][j]
        a.append(past[num-1])

        if ((i + 1)% 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h
    playing_stat['AM' + str(num)] = a

    return playing_stat

def add_form_df(playing_statistics):
    playing_statistics = add_form(playing_statistics,1)
    playing_statistics = add_form(playing_statistics,2)
    playing_statistics = add_form(playing_statistics,3)
    return playing_statistics

for i in range(len(playing_statistics)):
    playing_statistics[i] = add_form_df(playing_statistics[i])
playing_statistics[2].tail()
# 3.1.4 Join the game week feature (the first game week)
def get_mw(playing_stat):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

for i in range(len(playing_statistics)):
    playing_statistics[i] = get_mw(playing_statistics[i])

playing_statistics[2].tail()

# 3.1.5 Consolidation game information
playing_stat = pd.concat(playing_statistics, ignore_index=True)

cols = ['HTGD','ATGD','HTP','ATP']
playing_stat.MW = playing_stat.MW.astype(float)
for col in cols:
    playing_stat[col] = playing_stat[col] / playing_stat.MW

# View the last 5 data of the data set after constructing the feature
playing_stat.tail()
# 3.2 Delete some data
# Abandon the first three weeks of the game
playing_stat = playing_stat[playing_stat.MW > 3]
playing_stat.drop(['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'MW'],1, inplace=True)

playing_stat.keys()


# 3.3 Analyze the data we construct
# Total number of matches
n_matches = playing_stat.shape[0]

# Number of features
n_features = playing_stat.shape[1] - 1

# Number of home wins
n_homewins = len(playing_stat[playing_stat.FTR == 'H'])

# Proportion of home wins
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print("Total number of matches: {}".format(n_matches))
print("Total number of features: {}".format(n_features))
print("Home wins: {}".format(n_homewins))
print("Home win rate: {:.2f}%".format(win_rate))
# 3.4 Solve the problem of sample imbalance
# Define target, that is, whether to win at home
def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'
playing_stat['FTR'] = playing_stat.FTR.apply(only_hw)
# 3.5 Divide data into feature values and tag values
# Divide data into eigenvalues and tag values
X_all = playing_stat.drop(['FTR'],1)
y_all = playing_stat['FTR']
# Length of eigenvalue
len(X_all)
# 3.6 Data normalization, standardization
def convert_1(data):
    max=data.max()
    min=data.min()
    return (data-min)/(max-min)
r_data=convert_1(X_all['HTGD'])
# Data standardization
from sklearn.preprocessing import scale
cols = [['HTGD','ATGD','HTP','ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])
# 3.7 Convert feature data type
# Convert these features to a string type
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')

def preprocess_features(X):
    ''' Convert discrete type features to dummy coding features '''
    output = pd.DataFrame(index = X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
        output = output.join(col_data)
    return output

X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
'''
# Preview processed data
print("\nFeature values:")
#display(X_all.head())
'''
# 3.8 Pearson related heat map
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False
#Pearson heat map
# Map labels to 0 and 1
y_all=y_all.map({'NH':0,'H':1})
# Merge feature sets and labels
train_data=pd.concat([X_all,y_all],axis=1)

colormap = plt.cm.RdBu
plt.figure(figsize=(21,18))
plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_data.astype(float).corr(),linewidths=0.05,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=False)
plt.xticks(rotation=90)    

plt.yticks(rotation=360)

# Considering the characteristics of the sample set HTP and HTGD, the correlation between ATP and ATGD is over 90%
#So we remove the features HTP , ATP :X_all=X_all.drop(['HTP','ATP'],axis=1)
# 10 features most relevant to FTR
#FTR correlation matrix
plt.figure(figsize=(14,12))
k = 10 # number of variables for heatmap
cols = abs(train_data.astype(float).corr()).nlargest(k, 'FTR')['FTR'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, \
                 yticklabels=cols.values, xticklabels=cols.values)
plt.xticks(rotation=90)    

plt.yticks(rotation=360)
plt.show()

# 4. Establish a machine learning model and make predictions
# 4.1.2 Code Processing Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size = 0.3, \
                                                    random_state = 2,stratify = y_all)

# 4.3 Establishing a machine learning model and evaluating
# 4.3.1 Building a modelfrom time import time
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    ''' Training model '''
    # Record training duration
    start = time()
    print(start)
    clf.fit(X_train, y_train)
    end = time()
    print(end)
    print("Training time {0} seconds".format(end - start))
    return end - start
def predict_labels(clf, features, target):
    ''' Use models for prediction '''
    # Record forecast duration
    start = time()
    y_pred = clf.predict(features)
    end = time()
    '''
    print("Prediction time in {0} seconds".format(end - start))
    '''
    return f1_score(target, y_pred, pos_label=1), sum(target == y_pred) / float(len(y_pred)), end - start

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and evaluate the model '''
    # Indicate the classifier and the training set size
    print("Trained {} model，Number of samples {}。".format(clf.__class__.__name__, len(X_train)))
    # Training model
    timeTrain = train_classifier(clf, X_train, y_train)
    # Evaluate the model on the test set
    f1Train, accTrain ,timePTrain = predict_labels(clf, X_train, y_train)
    print("F1 scores and accuracy on the train set: {0} , {0}。".format(f1Train, accTrain))
    f1Test, accTest ,timePTest = predict_labels(clf, X_test, y_test)
    print("F1 scores and accuracy on the test set: {0} , {0}。".format(f1Test, accTest))
    return f1Train, accTrain ,timePTrain,f1Test, accTest ,timePTest ,timeTrain
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
# Create three models separately
f1Train = np.zeros((8,20,3))
accTrain = np.zeros((8,20,3))
timePTrain = np.zeros((8,20,3))
f1Test = np.zeros((8,20,3))
accTest = np.zeros((8,20,3))
timePTest = np.zeros((8,20,3))
timeTrain = np.zeros((8,20,3))
for i in range(0,8):
    for j in range(0,20):
        clf_A = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,fit_intercept=True, \
                                       intercept_scaling=1, class_weight=None, random_state=42,solver='liblinear', \
                                       max_iter=100, multi_class='ovr', verbose=0,warm_start=False, n_jobs=i+1)#(random_state = 42,n_job =-1)

        clf_B = RadiusNeighborsClassifier(radius=2.3, weights='uniform', algorithm='auto', \
                                              leaf_size=30, p=2, metric='minkowski', \
                                              outlier_label=None, metric_params=None, n_jobs=i+1)
        '''
            clf_B = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto',
                                 leaf_size=30, p=2, metric='minkowski', metric_params=None,
                                 n_jobs=-1)#clf_B = SVC(random_state = 42, kernel='rbf',gamma='auto')
        '''
        clf_C = xgb.XGBClassifier(nthread = i+1,seed = 42)

        
        f1Train[i][j][0], accTrain[i][j][0] ,timePTrain[i][j][0] ,f1Test[i][j][0] , accTest[i][j][0] ,timePTest[i][j][0] ,timeTrain[i][j][0] =  \
        train_predict(clf_A, X_train, y_train, X_test, y_test)
  
        f1Train[i][j][1], accTrain[i][j][1] ,timePTrain[i][j][1] ,f1Test[i][j][1] , accTest[i][j][1] ,timePTest[i][j][1] ,timeTrain[i][j][1] = \
        train_predict(clf_B, X_train, y_train, X_test, y_test)
        
        f1Train[i][j][2], accTrain[i][j][2] ,timePTrain[i][j][2] ,f1Test[i][j][2] , accTest[i][j][2] ,timePTest[i][j][2] ,timeTrain[i][j][2] = \
        train_predict(clf_C, X_train, y_train, X_test, y_test)
        print(i,j)
