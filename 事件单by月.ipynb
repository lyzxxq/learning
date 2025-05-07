#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#%% md
### 数据读取与预处理

- 读取 Excel 文件中的事件数据
- 事件类别分类（硬件故障与非硬件故障）
- 构造文本特征
- 使用 TF-IDF 进行文本特征提取
#%%
# 读取数据
train_data = pd.read_excel('D:/Files/事件单/2024年标题描述（二层）.xlsx')
test_data = pd.read_excel('D:/Files/事件单/25_01标题描述.xlsx')
required_columns = ['事件单号', '事件标题', '事件描述', '二层事件类别']
train_data = train_data.dropna(subset=required_columns)
test_data = test_data.dropna(subset=required_columns)

# 定义硬件故障类别
train_hardware_categories = ["服务器", "服务器-软件缺陷", "服务器-硬件问题", "伙伴侧基础设施问题", "非定位根因"]
train_data["分类类别"] = train_data["二层事件类别"].apply(lambda x: "硬件故障" if x in train_hardware_categories else "非硬件故障")

test_hardware_categories = ["服务器-软件缺陷", "服务器-硬件问题", "客户侧硬件设备问题", "非定位根因"]
test_data["分类类别"] = test_data["二层事件类别"].apply(lambda x: "硬件故障" if x in test_hardware_categories else "非硬件故障")

# 合并事件标题和事件描述
train_data['综合描述'] = train_data['事件标题'] + " " + train_data['事件描述']
test_data['综合描述'] = test_data['事件标题'] + " " + test_data['事件描述']

# 分割数据为特征和标签
X_train, y_train = train_data['综合描述'], train_data['分类类别']
X_test, y_test = test_data['综合描述'], test_data['分类类别']

# 文本特征提取（TF-IDF）
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
#%% md
### 类别不平衡处理
- 训练集过采样
#%%
# 训练集过采样
majority = train_data[train_data['分类类别'] == '硬件故障']
minority = train_data[train_data['分类类别'] == '非硬件故障']

# 过采样少数类别
minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

# 合并过采样后的数据集，并进行随机排序
train_balanced = pd.concat([majority, minority_upsampled])
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 提取X和y
X_train, y_train = train_balanced['综合描述'], train_balanced['分类类别']
X_test, y_test = test_data['综合描述'], test_data['分类类别']

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
#%% md
### Logistic Regression
#%%
# Logistic Regression
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
}

results = []
for penalty in param_grid['penalty']:
    for C in param_grid['C']:
        for solver in param_grid['solver']:
            # 条件判断，确保参数组合合法
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                continue
            if penalty == 'l2' and solver not in ['liblinear', 'saga', 'lbfgs', 'newton-cg']:
                continue
            model = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=42)
            try:
                model.fit(X_train_tfidf, y_train)
                y_pred = model.predict(X_test_tfidf)
                
                # 计算模型评估指标
                overall_accuracy = accuracy_score(y_test, y_pred)
                precision_hardware = precision_score(y_test, y_pred, pos_label='硬件故障')
                recall_hardware = recall_score(y_test, y_pred, pos_label='硬件故障')
                precision_nonhardware = precision_score(y_test, y_pred, pos_label='非硬件故障')
                recall_nonhardware = recall_score(y_test, y_pred, pos_label='非硬件故障')
                    
                # 记录结果
                results.append({
                    'penalty': penalty,
                    'C': C,
                    'solver': solver,
                    '整体准确率': overall_accuracy,
                    '硬件故障精准率': precision_hardware,
                    '硬件故障召回率': recall_hardware,
                    '非硬件故障精准率': precision_nonhardware,
                    '非硬件故障召回率': recall_nonhardware,
                })
                
                print(f"参数组合: penalty={penalty}, C={C}, solver={solver}, 整体准确率={overall_accuracy:.4f}")
                print(f"精准率（硬件故障）: {precision_hardware:.4f}, 召回率（硬件故障）: {recall_hardware:.4f}, 精准率（非硬件故障）: {precision_nonhardware:.4f}, 召回率（非硬件故障）: {recall_nonhardware:.4f}")
            except Exception as e:
                print(f"参数组合失败: penalty={penalty}, C={C}, solver={solver}, 错误={str(e)}")

results_df = pd.DataFrame(results)
results_df.to_excel('D:/Files/事件单/LR_1月_results.xlsx', index=False)

# 按准确率排序结果，并显示前5个最佳参数组合（如更多关注非硬件故障，可按照“非硬件故障召回率”排序）
sorted_results = results_df.sort_values(by='整体准确率', ascending=False)
print("前5个最佳参数组合:")
print(sorted_results.head())
#%% md
### Random Forest
#%%
param_grid_RF = {
    'n_estimators': [50, 100, 200],   # 决策树数量
    'max_depth': [None, 10, 20],      # 最大树深度
    'min_samples_split': [2, 5, 10]   # 最小分裂样本数
}

results_RF = []
for n_estimators in param_grid_RF['n_estimators']:
    for max_depth in param_grid_RF['max_depth']:
        for min_samples_split in param_grid_RF['min_samples_split']:
            # 初始化模型
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            try:
                
                model.fit(X_train_tfidf, y_train) 
                y_pred = model.predict(X_test_tfidf) 
                overall_accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label='硬件故障')
                recall = recall_score(y_test, y_pred, pos_label='硬件故障')
                precision2 = precision_score(y_test, y_pred, pos_label='非硬件故障')
                recall2 = recall_score(y_test, y_pred, pos_label='非硬件故障')
                    
                # 记录结果
                results_RF.append({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    '整体准确率': overall_accuracy,
                    '硬件故障精准率': precision,
                    '硬件故障召回率': recall,
                    '非硬件故障精准率': precision2,
                    '非硬件故障召回率': recall2,
                })
                
                print(f"参数组合: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, 整体准确率={overall_accuracy:.4f}")
                print(f"精准率（硬件故障）: {precision:.4f}, 召回率（硬件故障）: {recall:.4f}, 精准率（非硬件故障）: {precision2:.4f}, 召回率（非硬件故障）: {recall2:.4f}")
            except Exception as e:
                print(f"参数组合失败: penalty={penalty}, C={C}, solver={solver}, 错误={str(e)}")

results_df = pd.DataFrame(results_RF)
results_df.to_excel('D:/Files/事件单/RF_1月_results.xlsx', index=False)

sorted_results = results_df.sort_values(by='整体准确率', ascending=False)
print("前5个最佳参数组合:")
print(sorted_results.head())
#%% md
### SVM
#%%

param_grid_SVM = {
    'C': [0.1, 1, 10],              # 惩罚参数
    'kernel': ['linear', 'rbf'],    # 核函数
    'gamma': ['scale', 0.1, 1]      # 核函数系数
}

results_SVM = []
for C in param_grid_SVM['C']:
    for kernel in param_grid_SVM['kernel']:
        for gamma in param_grid_SVM['gamma']:
            # 初始化模型
            model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)
            try:
                
                model.fit(X_train_tfidf, y_train) 
                y_pred = model.predict(X_test_tfidf) 
                overall_accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label='硬件故障')
                recall = recall_score(y_test, y_pred, pos_label='硬件故障')
                precision2 = precision_score(y_test, y_pred, pos_label='非硬件故障')
                recall2 = recall_score(y_test, y_pred, pos_label='非硬件故障')
                    
                # 记录结果
                results_SVM.append({
                    'C': C,
                    'kernel': kernel,
                    'gamma': gamma,
                    '整体准确率': overall_accuracy,
                    '硬件故障精准率': precision,
                    '硬件故障召回率': recall,
                    '非硬件故障精准率': precision2,
                    '非硬件故障召回率': recall2,
                })
                print(f"参数组合: C={C}, kernel={kernel}, gamma={gamma}, 整体准确率={overall_accuracy:.4f}")
                print(f"精准率（硬件故障）: {precision:.4f}, 召回率（硬件故障）: {recall:.4f}, 精准率（非硬件故障）: {precision2:.4f}, 召回率（非硬件故障）: {recall2:.4f}")
                
            except Exception as e:
                print(f"参数组合失败: C={C}, kernel={kernel}, gamma={gamma}, 错误={str(e)}")

results_df = pd.DataFrame(results_SVM)
results_df.to_excel('D:/Files/事件单/SVM_1月_results.xlsx', index=False)

sorted_results = results_df.sort_values(by='整体准确率', ascending=False)
print("前5个最佳参数组合:")
print(sorted_results.head())
#%% md
### Naive Bayes
#%%
# 朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB

param_grid_NB = {
    'alpha': [0.1, 0.5, 1.0, 2.0],  # 平滑参数
    'fit_prior': [True, False]      # 是否学习类别的先验概率
}

results_NB = []
for alpha in param_grid_NB['alpha']:
    for fit_prior in param_grid_NB['fit_prior']:
        model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        try: 
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            
            # 计算指标
            overall_accuracy = accuracy_score(y_test, y_pred)
            hardware_precision = precision_score(y_test, y_pred, pos_label='硬件故障')
            hardware_recall = recall_score(y_test, y_pred, pos_label='硬件故障')
            nonhardware_precision = precision_score(y_test, y_pred, pos_label='非硬件故障')
            nonhardware_recall = recall_score(y_test, y_pred, pos_label='非硬件故障')
            
            cls_report = classification_report(
                y_test, y_pred, target_names=['硬件故障','非硬件故障'], output_dict=True
            )
            
            # 记录结果
            results_NB.append({
                'alpha': alpha,
                'fit_prior': fit_prior,
                '整体准确率': overall_accuracy,
                '硬件故障精准率': hardware_precision,
                '硬件故障召回率': hardware_recall,
                '非硬件故障精准率': nonhardware_precision,
                '非硬件故障召回率': nonhardware_recall,
            })
            
            print(f"参数组合: alpha={alpha}, fit_prior={fit_prior}, 整体准确率={overall_accuracy:.4f}")

            print(f"精准率（硬件故障）: {precision:.4f}, 召回率（硬件故障）: {recall:.4f}, 精准率（非硬件故障）: {precision2:.4f}, 召回率（非硬件故障）: {recall2:.4f}")
        except Exception as e:
            print(f"参数组合失败: alpha={alpha}, fit_prior={fit_prior}, 错误={str(e)}")


results_df = pd.DataFrame(results_NB)
results_df.to_excel('D:/Files/事件单/NBNBBB_1月_results.xlsx', index=False)

sorted_results = results_df.sort_values(by='整体准确率', ascending=False)

print("前5个最佳参数组合:")
print(sorted_results.head())
#%% md
### GBDT
#%%
# 梯度提升树
from sklearn.ensemble import GradientBoostingClassifier

param_grid_GB = {
    'n_estimators': [50, 100, 200],  # 弱学习器数量
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'max_depth': [3, 5, 10]  # 最大树深度
}

results_GB = []


for n_estimators in param_grid_GB['n_estimators']:
    for learning_rate in param_grid_GB['learning_rate']:
        for max_depth in param_grid_GB['max_depth']:

            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
            
            try:
                model.fit(X_train_tfidf, y_train)
                y_pred = model.predict(X_test_tfidf)
                
                # 计算指标
                overall_accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label='硬件故障')
                recall = recall_score(y_test, y_pred, pos_label='硬件故障')
                precision2 = precision_score(y_test, y_pred, pos_label='非硬件故障')
                recall2 = recall_score(y_test, y_pred, pos_label='非硬件故障')
                
                # 记录结果
                results_GB.append({
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    '整体准确率': overall_accuracy,
                    '硬件故障精准率': precision,
                    '硬件故障召回率': recall,
                    '非硬件故障精准率': precision2,
                    '非硬件故障召回率': recall2,
                })
                
                print(f"参数组合: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, 整体准确率={overall_accuracy:.4f}")
                print(f"精准率（硬件故障）: {precision:.4f}, 召回率（硬件故障）: {recall:.4f}, 精准率（非硬件故障）: {precision2:.4f}, 召回率（非硬件故障）: {recall2:.4f}")
            except Exception as e:
                print(f"参数组合失败: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, 错误={str(e)}")


results_df = pd.DataFrame(results_GB)
results_df.to_excel('D:/Files/事件单/GB_1月_results.xlsx', index=False)
sorted_results = results_df.sort_values(by='整体准确率', ascending=False)

print("前5个最佳参数组合:")
print(sorted_results.head())
#%% md
### AdaBoost
#%%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1],
    'estimator': [DecisionTreeClassifier(max_depth=d) for d in [1, 3, 5]]  # 直接传入决策树对象
}
results_AdaBoost = []
for n_estimators in param_grid['n_estimators']:
    for learning_rate in param_grid['learning_rate']:
        for estimator in param_grid['estimator']:
            model = AdaBoostClassifier(
                estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                algorithm='SAMME',
                random_state=42
            )
            
            try:
                model.fit(X_train_tfidf, y_train)
                y_pred = model.predict(X_test_tfidf)
                overall_accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label='硬件故障')
                recall = recall_score(y_test, y_pred, pos_label='硬件故障')
                precision2 = precision_score(y_test, y_pred, pos_label='非硬件故障')
                recall2 = recall_score(y_test, y_pred, pos_label='非硬件故障')
                    
                # 记录结果
                results_AdaBoost.append({
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'estimator_max_depth': estimator.max_depth,  # 直接获取决策树的 max_depth
                    '整体准确率': overall_accuracy,
                    '硬件故障精准率': precision,
                    '硬件故障召回率': recall,
                    '非硬件故障精准率': precision2,
                    '非硬件故障召回率': recall2,
                })

                
                print(f"参数组合: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={estimator.max_depth}, "
                      f"整体准确率={overall_accuracy:.4f}")
                print(f"精准率（硬件故障）: {precision:.4f}, 召回率（硬件故障）: {recall:.4f}, 精准率（非硬件故障）: {precision2:.4f}, 召回率（非硬件故障）: {recall2:.4f}")
            except Exception as e:
                print(f"参数组合失败: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={estimator.max_depth}, "
                      f"错误={str(e)}")

results_df = pd.DataFrame(results_AdaBoost)
results_df.to_excel('D:/Files/事件单/AB_1月_results.xlsx', index=False)
sorted_results = results_df.sort_values(by='整体准确率', ascending=False)
print("前5个最佳参数组合:")
print(sorted_results.head())

#%% md
### XGBoost
#%%
# XGBoost
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier  # Import XGBoost

# 标签编码：将分类变量转换为数值
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'colsample_bytree': [0.5, 0.8, 1],
    'subsample': [0.8, 1]
}

results_XGBoost = []
for max_depth in param_grid['max_depth']:
    for learning_rate in param_grid['learning_rate']:
        for n_estimators in param_grid['n_estimators']:
            for colsample_bytree in param_grid['colsample_bytree']:
                for subsample in param_grid['subsample']:
                    model = XGBClassifier(
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimators,
                        colsample_bytree=colsample_bytree,
                        subsample=subsample,
                        random_state=42
                    )
                    try:
                        model.fit(X_train_tfidf, y_train_encoded)
                        y_pred_encoded = model.predict(X_test_tfidf)

                        y_pred = label_encoder.inverse_transform(y_pred_encoded)
                        overall_accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, pos_label='硬件故障')
                        recall = recall_score(y_test, y_pred, pos_label='硬件故障')
                        precision2 = precision_score(y_test, y_pred, pos_label='非硬件故障')
                        recall2 = recall_score(y_test, y_pred, pos_label='非硬件故障')

                        results_XGBoost.append({
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'n_estimators': n_estimators,
                            'colsample_bytree': colsample_bytree,
                            'subsample': subsample,
                            '整体准确率': overall_accuracy,
                            '硬件故障精准率': precision,
                            '硬件故障召回率': recall,
                            '非硬件故障精准率': precision2,
                            '非硬件故障召回率': recall2,
                        })
                        print(f"参数组合: max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, "
                              f"colsample_bytree={colsample_bytree}, subsample={subsample}, 整体准确率={overall_accuracy:.4f}")
                        print(f"精准率（硬件故障）: {precision:.4f}, 召回率（硬件故障）: {recall:.4f}, 精准率（非硬件故障）: {precision2:.4f}, 召回率（非硬件故障）: {recall2:.4f}")
                
                    except Exception as e:
                        print(f"参数组合失败: max_depth={max_depth}, learning_rate={learning_rate}, "
                              f"n_estimators={n_estimators}, colsample_bytree={colsample_bytree}, subsample={subsample}, 错误={str(e)}")

results_df = pd.DataFrame(results_XGBoost)
results_df.to_excel('D:/Files/事件单/XGB_1月_results.xlsx', index=False)
sorted_results = results_df.sort_values(by='整体准确率', ascending=False)

print("前5个最佳参数组合:")
print(sorted_results.head())

#%%
