import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
import numpy as np


COL_INPUT = None
COL_Y = None
btn_predict = None
vars = None
lsvc = None
adab = None

st.title("EEEE")
st.title('AI for XXX')  # 算法名称 and XXX

def model_predit(model_name, model, input):
    y_pred_probas = []
    if isinstance(model, MultiOutputClassifier):
        for clf in model.estimators_:
            y_pred_proba = clf.predict_proba(input)[:, 1]
            y_pred_probas.append(y_pred_proba)
    elif model_name == "MLP":
        y_pred_proba_eph = model.predict_proba(input)[:, 0]
        y_pred_probas.append(y_pred_proba_eph)
        y_pred_proba_phe = model.predict_proba(input)[:, 1]
        y_pred_probas.append(y_pred_proba_phe)
        y_pred_proba_epi = model.predict_proba(input)[:, 2]
        y_pred_probas.append(y_pred_proba_epi)
    else:
        y_pred_proba_eph = model.predict_proba(input)[0][:, 1]
        y_pred_proba_phe = model.predict_proba(input)[1][:, 1]
        y_pred_proba_epi = model.predict_proba(input)[2][:, 1]
        y_pred_probas.append(y_pred_proba_eph)
        y_pred_probas.append(y_pred_proba_phe)
        y_pred_probas.append(y_pred_proba_epi)

    return y_pred_probas

def model_fit_score(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_probas = []

    if isinstance(model, MultiOutputClassifier):
        for clf in model.estimators_:
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            y_pred_probas.append(y_pred_proba)
    elif model_name == "MLP":
        y_pred_proba_eph = model.predict_proba(X_test)[:, 0]
        y_pred_probas.append(y_pred_proba_eph)
        y_pred_proba_phe = model.predict_proba(X_test)[:, 1]
        y_pred_probas.append(y_pred_proba_phe)
        y_pred_proba_epi = model.predict_proba(X_test)[:, 2]
        y_pred_probas.append(y_pred_proba_epi)
    else:
        y_pred_proba_eph = model.predict_proba(X_test)[0][:, 1]
        y_pred_proba_phe = model.predict_proba(X_test)[1][:, 1]
        y_pred_proba_epi = model.predict_proba(X_test)[2][:, 1]
        y_pred_probas.append(y_pred_proba_eph)
        y_pred_probas.append(y_pred_proba_phe)
        y_pred_probas.append(y_pred_proba_epi)
    eph = {}
    phe = {}
    epi = {}
    eph_ = {}
    phe_ = {}
    epi_ = {}
    eph['accuracy'] = accuracy_score(y_test['intraop_eph'], y_pred[:, 0])
    eph['f1'] = f1_score(y_test['intraop_eph'], y_pred[:, 0])
    eph['precision'] = precision_score(y_test['intraop_eph'], y_pred[:, 0])
    eph['recall'] = recall_score(y_test['intraop_eph'], y_pred[:, 0])
    eph_fpr, eph_tpr, _ = roc_curve(y_test['intraop_eph'], y_pred_probas[0])
    eph['auc'] = auc(eph_fpr, eph_tpr)
    eph_['fpr'] = eph_fpr
    eph_['tpr'] = eph_tpr

    phe['accuracy'] = accuracy_score(y_test['intraop_phe'], y_pred[:, 1])
    phe['f1'] = f1_score(y_test['intraop_phe'], y_pred[:, 1])
    phe['precision'] = precision_score(y_test['intraop_phe'], y_pred[:, 1])
    phe['recall'] = recall_score(y_test['intraop_phe'], y_pred[:, 1])
    phe_fpr, phe_tpr, _ = roc_curve(y_test['intraop_phe'], y_pred_probas[1])
    phe['auc'] = auc(phe_fpr, phe_tpr)
    phe_['fpr'] = phe_fpr
    phe_['tpr'] = phe_tpr

    epi['accuracy'] = accuracy_score(y_test['intraop_epi'], y_pred[:, 2])
    epi['f1'] = f1_score(y_test['intraop_epi'], y_pred[:, 2])
    epi['precision'] = precision_score(y_test['intraop_epi'], y_pred[:, 2])
    epi['recall'] = recall_score(y_test['intraop_epi'], y_pred[:, 2])
    epi_fpr, epi_tpr, _ = roc_curve(y_test['intraop_epi'], y_pred_probas[2])
    epi['auc'] = auc(epi_fpr, epi_tpr)
    epi_['fpr'] = epi_fpr
    epi_['tpr'] = epi_tpr
    return eph, phe, epi, eph_, phe_, epi_

def process_data(data, train_state=True):
    proc_data = pd.DataFrame(columns=data.columns)

    for index, case_data in data.iterrows():
        age = case_data['age']
        bmi = case_data['bmi']
        asa = case_data['asa']
        preop_hb = case_data['preop_hb']

        sex = case_data['sex']
        if sex.lower() == 'm':
            sex = 1
        else:
            sex = 0      

        emop = case_data['emop']
        if emop.lower() == 'n':
            emop = 0
        else:
            emop = 1
        
        optype = case_data['optype']
        if optype.lower() == 'colorectal':
            optype  = 0
        elif optype.lower() == 'stomach':
            optype  = 1
        elif optype.lower() == 'vascular':
            optype  = 2
        elif optype.lower() == 'transplantation':
            optype  = 3
        elif optype.lower() == 'minor resection':
            optype  = 4
        elif optype.lower() == 'hepatic':
            optype  = 5
        elif optype.lower() == 'biliary/pancreas':
            optype  = 6
        elif optype.lower() == 'major resection':
            optype  = 7
        elif optype.lower() == 'thyroid':
            optype  = 8
        elif optype.lower() == 'breast':
            optype  = 9
        else:
            optype  = 10

        approach = case_data['approach']
        if approach.lower() == 'open':
            approach = 0
        elif approach.lower() == 'videoscopic':
            approach = 1
        elif approach.lower() == 'robotic':
            approach =2

        preop_dm = case_data['preop_dm']
        if preop_dm.lower() == 'n':
            preop_dm = 0
        else:
            preop_dm = 1

        preop_htn = case_data['preop_htn']
        if preop_htn.lower() == 'n':
            preop_htn = 0
        else:
            preop_htn = 1
        
        preop_pft = case_data['preop_pft']
        if preop_pft.lower() == 'normal':
            preop_pft = 0
        else:
            preop_pft = 1

        if train_state:
            intraop_eph = case_data['intraop_eph']
            if intraop_eph == 0:
                intraop_eph = 0
            else:
                intraop_eph = 1
            intraop_phe = case_data['intraop_phe']
            if intraop_phe == 0:
                intraop_phe = 0
            else:
                intraop_phe = 1
            intraop_epi = case_data['intraop_epi']
            if intraop_epi == 0:
                intraop_epi = 0
            else:
                intraop_epi = 1

            proc_data.loc[len(proc_data)] = (case_data["caseid"], intraop_eph, 
                                            intraop_phe, intraop_epi, age, sex, bmi, asa, 
                                            emop, optype, preop_dm, approach, preop_htn, 
                                            preop_pft, preop_hb)
        else:
            proc_data.loc[len(proc_data)] = (age, sex, bmi, asa, 
                                            emop, optype, preop_dm, approach, preop_htn, 
                                            preop_pft, preop_hb)
    return proc_data
def do_processing():
    global vars
    global lre, lda, ada, mlp
    global COL_INPUT
    data = pd.read_csv("mydata2.csv")

    proc_data = process_data(data)
    # 预处理数据
    
    st.subheader("Dataset Description")

    st.write(proc_data.describe())

    if st.checkbox('Show detail of this dataset'):
        st.write(data)

    scaler = StandardScaler()
    mydata = proc_data
    mydata = pd.get_dummies(proc_data, columns=['sex', 'emop', 'optype', 'preop_dm', 'approach', 'preop_htn', 'preop_pft'])
    mydata[['age', 'bmi', 'asa', 'preop_hb']] = scaler.fit_transform(proc_data[['age', 'bmi', 'asa', 'preop_hb']])

    # 分割特征和目标变量
    X = mydata.drop(columns=['caseid', 'intraop_eph', 'intraop_phe', 'intraop_epi'])
    COL_INPUT = pd.DataFrame(columns=X.columns)
    y = mydata[['intraop_eph', 'intraop_phe', 'intraop_epi']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # LogisticRegression
    lre = LogisticRegression()
    lre = MultiOutputClassifier(lre)

    # AdaBoost
    ada = AdaBoostClassifier(algorithm='SAMME', n_estimators=10, random_state=42, estimator=DecisionTreeClassifier(max_depth=2, random_state=42, criterion='gini'))
    ada = MultiOutputClassifier(ada)

    # 线性判别分析 (Linear Discriminant Analysis, LDA)
    lda = LinearDiscriminantAnalysis()
    lda = MultiOutputClassifier(lda)

    # MLP
    mlp = MLPClassifier(activation='logistic', learning_rate='adaptive', solver='adam', max_iter=500, hidden_layer_sizes=[64, 32, 16], random_state=42)
    
    col1, col2 = st.columns(2)

    with st.spinner("Training, please wait..."):
        lre_eph, lre_phe, lre_epi, lre_eph_, lre_phe_, lre_epi_  = model_fit_score("LogisticRegression", lre, X_train, X_test, y_train, y_test)
        ada_eph, ada_phe, ada_epi, ada_eph_, ada_phe_, ada_epi_ = model_fit_score("AdaBoost", ada, X_train, X_test, y_train, y_test)
        lda_eph, lda_phe, lda_epi, lda_eph_, lda_phe_, lda_epi_ = model_fit_score("LDA", lda, X_train, X_test, y_train, y_test)
        mlp_eph, mlp_phe, mlp_epi, mlp_eph_, mlp_phe_, mlp_epi_ = model_fit_score("MLP", mlp, X_train, X_test, y_train, y_test)

    with col1:
        st.subheader("LogisticRegression")
        st.text("The prediction result of intraop_eph")
        st.write(lre_eph)
        st.text("The prediction result of intraop_phe")
        st.write(lre_phe)
        st.text("The prediction result of intraop_epi")
        st.write(lre_epi)

        st.subheader("LDA")
        st.text("The prediction result of intraop_eph")
        st.write(lda_eph)
        st.text("The prediction result of intraop_phe")
        st.write(lda_phe)
        st.text("The prediction result of intraop_epi")
        st.write(lda_epi)
        
    with col2:
        st.subheader("AdaBoost")
        st.text("The prediction result of intraop_eph")
        st.write(ada_eph)
        st.text("The prediction result of intraop_phe")
        st.write(ada_phe)
        st.text("The prediction result of intraop_epi")
        st.write(ada_epi)
    
        st.subheader("MLP")
        st.text("The prediction result of intraop_eph")
        st.write(mlp_eph)
        st.text("The prediction result of intraop_phe")
        st.write(mlp_phe)
        st.text("The prediction result of intraop_epi")
        st.write(mlp_epi)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        plt.figure(figsize=(4, 4))
        plt.plot(lre_eph_['fpr'], lre_eph_['tpr'], label=f'LRE, AUC={lre_eph['auc']:.3f}')
        plt.plot(lda_eph_['fpr'], lda_eph_['tpr'], label=f'LDA, AUC={lda_eph['auc']:.3f}')
        plt.plot(ada_eph_['fpr'], ada_eph_['tpr'], label=f'ADA, AUC={ada_eph['auc']:.3f}')
        plt.plot(mlp_eph_['fpr'], mlp_eph_['tpr'], label=f'MLP, AUC={mlp_eph['auc']:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')

        # 设置图像参数
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC - intraop_eph')
        plt.legend(loc="lower right")
        st.pyplot(plt)
    with c2:
        plt.figure(figsize=(4, 4))
        plt.plot(lre_phe_['fpr'], lre_phe_['tpr'], label=f'LRE, AUC={lre_phe['auc']:.3f}')
        plt.plot(lda_phe_['fpr'], lda_phe_['tpr'], label=f'LDA, AUC={lda_phe['auc']:.3f}')
        plt.plot(ada_phe_['fpr'], ada_phe_['tpr'], label=f'ADA, AUC={ada_phe['auc']:.3f}')
        plt.plot(mlp_phe_['fpr'], mlp_phe_['tpr'], label=f'MLP, AUC={mlp_phe['auc']:.3f}')
        # 设置图像参数
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC - intraop_phe')
        plt.legend(loc="lower right")
        st.pyplot(plt)
    with c3:
        plt.figure(figsize=(4, 4))
        plt.plot(lre_epi_['fpr'], lre_epi_['tpr'], label=f'LRE, AUC={lre_epi['auc']:.3f}')
        plt.plot(lda_epi_['fpr'], lda_epi_['tpr'], label=f'LDA, AUC={lda_epi['auc']:.3f}')
        plt.plot(ada_epi_['fpr'], ada_epi_['tpr'], label=f'ADA, AUC={ada_epi['auc']:.3f}')
        plt.plot(mlp_epi_['fpr'], mlp_epi_['tpr'], label=f'MLP, AUC={mlp_epi['auc']:.3f}')
        # 设置图像参数
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC - intraop_epi')
        plt.legend(loc="lower right")
        st.pyplot(plt)


def do_predict():
    global vars
    global lre, lda, ada, mlp
    
    scaler = StandardScaler()
    mydata = pd.read_csv("mydata2.csv")
    mydata = mydata.drop(columns=['caseid', 'intraop_eph', 'intraop_phe', 'intraop_epi'])
    # print('before:', len(mydata))

    mydata.loc[len(mydata)] = vars
    proc_data = process_data(mydata, train_state=False)
    # print('after:', len(mydata))
    input = proc_data
    st.text("Preview for detail of this predict data")
    st.write(input.iloc[-1])
    # print('before:', input.head)
    input = pd.get_dummies(proc_data, columns=['sex', 'emop', 'optype', 'preop_dm', 'approach', 'preop_htn', 'preop_pft'])
    input[['age', 'bmi', 'asa', 'preop_hb']] = scaler.fit_transform(proc_data[['age', 'bmi', 'asa', 'preop_hb']])
    # print('after:', input.head)
    # print(input.tail(1))
    
    lre_res = model_predit("LogisticRegression", lre, input.tail(1))
    lda_res = model_predit("LDA", lda, input.tail(1))
    ada_res = model_predit("AdaBoost", ada, input.tail(1))
    mlp_res = model_predit("MLP", mlp, input.tail(1))

    # st.markdown(r"$\color{red}{The prediction results of model LRE for intraop_eph/phe/epi are}$))
    st.markdown(f":rainbow[The prediction results of model LRE for intraop_eph/phe/epi are:({np.round(lre_res[0], 3)}, {np.round(lre_res[1], 3)}, {np.round(lre_res[2], 3)})]")
    st.markdown(f":rainbow[The prediction results of model LDA for intraop_eph/phe/epi are:({np.round(lda_res[0], 3)}, {np.round(lda_res[1], 3)}, {np.round(lda_res[2], 3)})]")
    st.markdown(f":rainbow[The prediction results of model ADA for intraop_eph/phe/epi are:({np.round(ada_res[0], 3)}, {np.round(ada_res[1], 3)}, {np.round(ada_res[2], 3)})]")
    st.markdown(f":rainbow[The prediction results of model MLP for intraop_eph/phe/epi are:({np.round(mlp_res[0], 3)}, {np.round(mlp_res[1], 3)}, {np.round(mlp_res[2], 3)})]")



def setup_selectors():
    global vars, btn_predict
    cols = st.columns(2)
    with cols[0]:

        age = st.slider("age", 0, 100)
        bmi = st.slider('bmi', 0, 50)
        asa = st.slider("asa", 0, 4)
        preop_hb = st.slider("preophb", 0, 100)
        sex = st.radio("sex", ["M", "F"])
        emop = st.radio("emop", ["N", "Y"])
        
    with cols[1]:
        preop_dm = st.radio("preop_dm", ["N", "Y"])
        preop_htn = st.radio("preop_htn", ["N", "Y"])
        preop_pft = st.radio("preop_pft", ["Normal", "Others"])
        approach = st.radio("approach", ["open", "videoscopic", "robotic"])
        optype = st.radio("optype", ["colorectal", "stomach", "vascular", "transplantation", "minor resection",
                                    "hepatic", "biliary/pancreas", "major resection", "thyroid", "breast", "others"
                                    ])

    vars = {"age": age, "sex":sex, "bmi":bmi, "asa":asa, "emop":emop, "optype":optype, "preop_dm":preop_dm,
            "approach":approach, "preop_htn":preop_htn, "preop_pft":preop_pft, "preop_hb":preop_hb}
    

    with cols[0]:
        btn_predict = st.button("Do Predict")
    if btn_predict:
        do_predict()

if __name__ == "__main__":
    do_processing()
    setup_selectors()

            
