import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, \
    confusion_matrix, classification_report, balanced_accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate, GridSearchCV, \
    validation_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler, OneHotEncoder, \
    PolynomialFeatures, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz, export_text
from statsmodels.stats.proportion import proportions_ztest
import re
import graphviz
import pydotplus, skompiler, astor, joblib, warnings
import advanced_functional_era as afe

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 500)


df_ = pd.read_csv("datasets/Cardiovascular_Disease_Dataset.csv")
df = df_.copy()



afe.ilkbakis(df)
kategorik, sayisal, sayisal_kategorik, kategorik_kardinal, tarih = afe.degisken_siniflama(df)
afe.kategorik_ozet(df, cols=kategorik, plot=True)
afe.sayisal_ozet(df, sayisal, True)

for i in ['gender','chestpain','fastingbloodsugar','restingrelectro','exerciseangia','slope','noofmajorvessels']:
    df = df.astype({i: "category"})


df["age_new"] = pd.cut(df["age"], [df["age"].min(), 40, 60, df["age"].max()], labels=["young", "middle", "old"],
                       include_lowest=True)

yas_siralama = {'young': 0, 'middle': 1, 'old': 2}
df['age_new_encode'] = df['age_new'].map(yas_siralama)


for i in ['gender', 'chestpain','fastingbloodsugar', 'restingrelectro', 'exerciseangia', 'slope', 'noofmajorvessels']:
    df = afe.one_hot_encoder(df,i)





for i in sayisal:
    afe.outlier_kontrol(df, i)

df = df[df["serumcholestrol"]>0]


afe.eksik_veriye_bakis(df)

df[['restingBP', 'serumcholestrol', 'maxheartrate',"oldpeak","target"]].corr()




columns_list = [
    'restingBP',
    'serumcholestrol',
    'maxheartrate',
    'oldpeak',
    'age_new_encode',
    'gender_1',
    'chestpain_1',
    'chestpain_2',
    'chestpain_3',
    'fastingbloodsugar_1',
    'restingrelectro_1',
    'restingrelectro_2',
    'exerciseangia_1',
    'slope_1',
    'slope_2',
    'slope_3',
    'noofmajorvessels_1',
    'noofmajorvessels_2',
    'noofmajorvessels_3'
]

X = df[columns_list]
y = df["target"]


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)


standart = ['restingBP', 'serumcholestrol',"maxheartrate","oldpeak"] # standartlaştırma yapılacaklar

ss = StandardScaler()
X_train[standart] = ss.fit_transform(X_train[standart])
X_test[standart] = ss.transform(X_test[standart])


reg_model = LogisticRegression()
reg_model.fit(X_train,y_train)


y_pred = reg_model.predict(X_test)
y_train_pred = reg_model.predict(X_train)

y_prob = reg_model.predict_proba(X_test)[:, 1]




print(classification_report(y_test,y_pred))

confusion_matrix(y_test,y_pred)


print("\n--- Model Performansı (Eğitim Seti) ---")
print("Sınıflandırma Raporu:\n", classification_report(y_train, y_train_pred))
print("Karmaşıklık Matrisi:\n", confusion_matrix(y_train, y_train_pred))
print(f"ROC AUC Skoru: {roc_auc_score(y_train, y_train_pred):.4f}")


print("\n--- Model Performansı (Test Seti) ---")
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karmaşıklık Matrisi:\n", confusion_matrix(y_test, y_pred))
print(f"ROC AUC Skoru: {roc_auc_score(y_test, y_prob):.4f}")



########

roc_auc_score(y_test,y_prob)


fpr, tpr, thresholds = roc_curve(y_test,y_prob)

roc_auc = auc(fpr, tpr)



cm = confusion_matrix(y_test, y_pred)
score = reg_model.score(X_test,y_test)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels="0", yticklabels="1",
            linewidths=.5, linecolor='gray')

plt.title(f'Karmaşıklık Matrisi; Accuracy : {score:.2f}')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.show(block=True)



fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.50)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR) / Recall')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show(block=True)




