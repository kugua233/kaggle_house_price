import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from scipy import stats         #统计函数库
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

f1 = open(r'C:\Users\YC2Z\Desktop\算法比赛\Kaggle\波士顿房价预测\all\train.csv', encoding='utf-8')
f2 = open(r'C:\Users\YC2Z\Desktop\算法比赛\Kaggle\波士顿房价预测\all\test.csv', encoding='utf-8')
train = pd.read_csv(f1, sep=",")
test = pd.read_csv(f2, sep=",")
# =============================================================================

# 打乱顺序
# train = train.reindex(np.random.permutation(train.index))
# =============================================================================

# 检查数据
# =============================================================================
# 查看数据的数量和特征值的个数
print("The train data size before dropping Id feature is : {} ".format(
    train.shape))
print("The test data size before dropping Id feature is : {} ".format(
    test.shape))

# 前几行数据初始印象
# train.head()

# 了解数据分布情况 这个表展示的数据都只为数值型的数据
# print(train.describe())
# train.hist(bins=50, figsize=(30, 35))
# plt.show()

# 看看数据的统计是否有缺失值
# train.info()
# =============================================================================

# 拆分数据集
# =============================================================================

# 数据处理
# =============================================================================
# 1、整合数据：删ID列，保留train标签，合并train和test
train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis=1, inplace=True)  # axis=0，那么输出矩阵是1行;  axis=1，输出矩阵是1列
test.drop("Id", axis=1, inplace=True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(
    train.shape))
print(
    "The test data size after dropping Id feature is : {} ".format(test.shape))


# 对train处理离群值和对SalePrice进行对数处理
attributes = ["SalePrice", "1stFlrSF", "GarageYrBlt", "GrLivArea", "YearBuilt","TotalBsmtSF"]
scatter_matrix(train[attributes], figsize=(12, 8))
plt.show()

corrmat = train.corr()
plt.subplots(figsize=(10,10))
sns.heatmap(corrmat, vmax=0.9, square=True)   # 最大显示值

fig, ax = plt.subplots()     #建立画布，默认一幅图
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


train = train.drop(train[(train['SalePrice'] < 300000) & (train['1stFlrSF'] > 4000)].index)
train = train.drop(train[(train['SalePrice'] < 300000) & (train['GrLivArea'] > 4000)].index)
train['SalePrice'] = np.log1p(train['SalePrice'])
print(
    "The train data size after clean outlier is : {} ".format(train.shape))

n_train = train.shape[0]
n_test = test.shape[0]

#  使用concat拼接时要用reset_index，否则得到的index会重复，容易造成后续操作出错
y_train = train.SalePrice.values
full = pd.concat((train, test)).reset_index(drop=True)
full.drop(['SalePrice'], axis=1, inplace=True)
print("full size is : {}".format(full.shape))


# ===================================================================================================

# 2、清理缺失值，中位数均数填补 或者 更精确地映射到更高维度
# full_null = full.isnull().sum() / len(full) * 100
# print(full_null[full_null > 0].sort_values(ascending=False))   # 查看缺失情况 ，34个属性有缺失
'''
如果我们仔细观察一下data_description里面的内容的话，会发现很多缺失值都有迹可循，比如上表第一个PoolQC，
表示的是游泳池的质量，其值缺失代表的是这个房子本身没有游泳池，因此可以用 “None” 来填补

另外，比如 TotalBsmtSF 表示地下室的面积，如果一个房子本身没有地下室，则缺失值就用0来填补。（注意区分None和0的区别）

其余可用中位数或者平均数解决
'''
col_1 = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "GarageYrBlt", "GarageType", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]
for col in col_1:
    full[col].fillna("None", inplace=True)

col_2 = ["MasVnrArea", "BsmtUnfSF", "TotalBsmtSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1", "GarageArea"]
for col in col_2:
    full[col].fillna(0, inplace=True)

full.drop("LotFrontage", axis=1, inplace=True)    # 缺失率太高且为float类型直接去掉

# 对object/数值类型采用众数和中位数填补
col_3 = ["BsmtFullBath", "BsmtHalfBath"]
for col in col_3:
    median = full[col].median()
    full[col].fillna(median, inplace=True)

col_4 = ["MSZoning", "Utilities", "Functional", "Electrical", "Exterior1st", "Exterior2nd",  "KitchenQual", "SaleType"]
for col in col_4:
    full[col] = full[col].fillna(full[col].mode()[0])   # mode()  [0]对行取众数 [1]是对列取众数(这里没用到）


# 再次检查缺失情况
full_null = full.isnull().sum() / len(full) * 100
print(full_null[full_null > 0].sort_values(ascending=False))


# 再次检查缺失情况
# full_null = full.isnull().sum() / len(full) * 100
# print(full_null[full_null > 0].sort_values(ascending=False))


# 偏序用LabelEncoder 无偏序关系用OneHotEncoder
# 处理一些是数值但实际为分类的情况，例如数值表示的**等级，年月份
col_5 = ["MSSubClass", "OverallCond", "YrSold", "MoSold"]
for col in col_5:
    full[col] = full[col].apply(str)

# 使用 LabelEncoder 转换下述特征
col_6 = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
for col in col_6:
    lbl = LabelEncoder()
    lbl.fit(list(full[col].values))
    full[col] = lbl.transform(list(full[col].values))
print('After LabelEncoder transform: {}'.format(full.shape))


# 使用 OneHotEncoder 转换下述特征(只能用于数值型)，若处理字符串的文本型需要先用LabelEncoder，再用OneHotEncoder
# 注意fit_transform()用于 2D 数组，而原始数据是一个 1D 数组，所以需要将其变形：
# 我们这里用pd.get_dummies()，可以直接将string转为int类型的编码
# col_7 = ('MSSubClass', 'YrSold', 'MoSold')
# for col in col_7:
#     oht = OneHotEncoder(sparse=False)
#     full_cat = full[col]    # full[]为一维数组 full[[]]是带字段的二维
#     housing_cat_1hot = oht.fit_transform(full_cat.reshape(-1, 1))
#     print(housing_cat_1hot)
# print('After OneHotEncoder transform: {}'.format(full.shape))


# ==========================================================

# 增加更多重要的特征
# Adding total sqfootage feature 房价一般会考虑整个房子的总面积
full['TotalSF'] = full['TotalBsmtSF'] + full[
    '1stFlrSF'] + full['2ndFlrSF']
# full['new_th'] = full['YrSold'] - full['YearBuilt']

# Skewed features
numeric_feats = full.dtypes[full.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = full[numeric_feats].apply(               # DataFrame.skew()求偏度，直观看来就是密度函数曲线尾部的相对长度
    lambda x: skew(x.dropna())).sort_values(ascending=False)    # dataframe.dropna()可以按行丢弃带有nan的数据
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew': skewed_feats})
# print(skewness.head(10))

# Box Cox Transformation of (highly) skewed features
# We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+x .
# Note that setting  λ=0  is equivalent to log1p used above for the target variable.
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(
    skewness.shape[0]))

skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    # all_data[feat] += 1
    full[feat] = boxcox1p(full[feat], lam)      # boxcox1p为类似log的转换函数

# Getting dummy categorical features
all_data = pd.get_dummies(full)
print(all_data.shape)

# # Getting the new train and test sets.
train = all_data[:n_train]
test = all_data[n_train:]

# --------------------------------PCA降维------------------------------------------------
# scaler = RobustScaler()
# train = scaler.fit(train).transform(train)
# test = scaler.transform(test)
#
# pca = PCA(n_components=0.95)
# train = pca.fit_transform(X_scaled)
# test = pca.transform(test_scaled)
# --------------------------------PCA降维------------------------------------------------


# Validation function交叉验证
# Scikit-Learn 交叉验证cross_val_score功能期望的是效用函数（越大越好）而不是损失函数（越低越好），因此得分函数实际上与 MSE 相反（即负值）
n_folds = 5
def rmsle_cv(model):
    kf = KFold(     # 打乱数据后的5折交叉
        n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(
        model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    print("rmse", rmse)
    return (rmse)


# 模型
# make_pipeline函数是Pipeline类的简单实现，只需传入每个step的类实例即可，不需自己命名，自动将类的小写设为该step的名。
# LASSO Regression :
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))  # RobustScaler适用具有异常的数据，它会根据中位数或者四分位数去中心化数据。
# Elastic Net Regression
ENet = make_pipeline(
    RobustScaler(), ElasticNet(
        alpha=0.0005, l1_ratio=.9, random_state=3))
# Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# Gradient Boosting Regression
GBoost = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    max_features='sqrt',
    min_samples_leaf=15,
    min_samples_split=10,
    loss='huber',
    random_state=5)
#  XGboost
model_xgb = xgb.XGBRegressor(
    colsample_bytree=0.4603,
    gamma=0.0468,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1.7817,
    n_estimators=2200,
    reg_alpha=0.4640,
    reg_lambda=0.8571,
    subsample=0.5213,
    silent=1,
    random_state=7,
    nthread=-1)
#  lightGBM
model_lgb = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11)
# Base models scores
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print(
    "Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(),
                                                          score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# 两个模型融合：平均法和学习法Stacking
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):    # X,y 为train样本和标签
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):   # X为测试样本
        predictions = np.column_stack(
            [model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)


# 评价这四个模型用平均法的的好坏
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(),
                                                              score.std()))


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]  # 用list()
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))     # 初始化初级学习器的输出
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):     #  split(X, y=None, groups=None) 将数据集划分成训练集和验证集，返回索引生成器
                instance = clone(model)
                self.base_models_[i].append(instance)   # 原模型再加入新克隆模型，用于预测
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])     # 训练集fit,用留出的验证集进行预测出结果
                out_of_fold_predictions[holdout_index, i] = y_pred  # 保留输出数据，用于次级学习器

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    # meta_features为stacking的预测数据，其值等于原预测数据在模型上的预测平均值
    def predict(self, X):
        meta_features = np.column_stack([       # np.column_stack矩阵按列合并
            np.column_stack([model.predict(X) for model in base_models]).mean(
                axis=1) for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(
    base_models=(ENet, GBoost, KRR), meta_model=lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(),
                                                               score.std()))


# define a rmsle evaluation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))   # 平均平方误差


# Final Training and Prediction
# StackedRegressor


stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))   #   expm1等于exp(x)-1;将预测值恢复成没有log时的状态
print(rmsle(y_train, stacked_train_pred))

# XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
# lightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test))
print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train, stacked_train_pred * 0.70 + xgb_train_pred * 0.15 +
            lgb_train_pred * 0.15))
# 模型融合的预测效果
ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

# 保存结果
result = pd.DataFrame()
result['Id'] = test_ID
result['SalePrice'] = ensemble

# q1 = result['SalePrice'].quantile(0.0042)
# q2 = result['SalePrice'].quantile(0.99)
#
# result['SalePrice'] = result['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
# result['SalePrice'] = result['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
# index=False 是用来除去行编号
result.to_csv(r'C:\Users\YC2Z\Desktop\house_price\result.csv', index=False)
print('##########结束训练##########')