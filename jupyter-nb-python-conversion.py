# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('laptop_data.csv')

# %%
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.duplicated().sum()

# %%
df.isnull().sum()

# %%
df.drop(columns=['Unnamed: 0'],inplace=True)

# %%
df.head()

# %%
df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

# %%
df.head()

# %%
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

# %%
df.info()

# %%
import seaborn as sns

# %%
sns.distplot(df['Price'])

# %%
df['Company'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
df['TypeName'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
sns.distplot(df['Inches'])

# %%
sns.scatterplot(x=df['Inches'],y=df['Price'])

# %%
df['ScreenResolution'].value_counts()

# %%
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

# %%
df.sample(5)

# %%
df['Touchscreen'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Touchscreen'],y=df['Price'])

# %%
df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

# %%
df.head()

# %%
df['Ips'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Ips'],y=df['Price'])

# %%
new = df['ScreenResolution'].str.split('x',n=1,expand=True)

# %%
df['X_res'] = new[0]
df['Y_res'] = new[1]

# %%
df.sample(5)

# %%
df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

# %%
df.head()

# %%
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')

# %%
df.info()

# %%
df.corr(numeric_only=True)['Price']


# %%
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

# %%
df.corr(numeric_only=True)['Price']


# %%
df.drop(columns=['ScreenResolution'],inplace=True)

# %%
df.head()

# %%
df.drop(columns=['Inches','X_res','Y_res'],inplace=True)

# %%
df.head()

# %%
df['Cpu'].value_counts()

# %%
df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))

# %%
df.head()

# %%
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

# %%
df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)

# %%
df.head()

# %%
df['Cpu brand'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
df.drop(columns=['Cpu','Cpu Name'],inplace=True)

# %%
df.head()

# %%
df['Ram'].value_counts().plot(kind='bar')

# %%
sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
df['Memory'].value_counts()

# %%
# Step 1: Clean Memory column
df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '', regex=False)
df['Memory'] = df['Memory'].str.replace('TB', '000', regex=False)

# Step 2: Split into first and second layer
mem_split = df['Memory'].str.split('+', n=1, expand=True)
df['first'] = mem_split[0].str.strip()
df['second'] = mem_split[1].fillna("0").str.strip()

# Step 3: Extract storage types from both parts
df['Layer1HDD'] = df['first'].str.contains('HDD').astype(int)
df['Layer1SSD'] = df['first'].str.contains('SSD').astype(int)
df['Layer1Hybrid'] = df['first'].str.contains('Hybrid').astype(int)
df['Layer1Flash_Storage'] = df['first'].str.contains('Flash Storage').astype(int)

df['Layer2HDD'] = df['second'].str.contains('HDD').astype(int)
df['Layer2SSD'] = df['second'].str.contains('SSD').astype(int)
df['Layer2Hybrid'] = df['second'].str.contains('Hybrid').astype(int)
df['Layer2Flash_Storage'] = df['second'].str.contains('Flash Storage').astype(int)

# Step 4: Remove non-digit characters (only extract the numeric part)
df['first'] = df['first'].str.extract('(\d+)')
df['second'] = df['second'].str.extract('(\d+)')

# Step 5: Convert to int
df['first'] = df['first'].fillna(0).astype(int)
df['second'] = df['second'].fillna(0).astype(int)

# Step 6: Create final features
df['HDD'] = df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD']
df['SSD'] = df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD']
df['Hybrid'] = df['first'] * df['Layer1Hybrid'] + df['second'] * df['Layer2Hybrid']
df['Flash_Storage'] = df['first'] * df['Layer1Flash_Storage'] + df['second'] * df['Layer2Flash_Storage']

# Step 7: Drop temp columns
df.drop(columns=[
    'first', 'second',
    'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
    'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'
], inplace=True)


# %%
df.sample(5)

# %%
df.drop(columns=['Memory'],inplace=True)

# %%
df.head()

# %%
df.corr(numeric_only=True)['Price']


# %%
df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)

# %%
df.head()

# %%
df['Gpu'].value_counts()

# %%
df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])

# %%
df.head()

# %%
df['Gpu brand'].value_counts()

# %%
df = df[df['Gpu brand'] != 'ARM']

# %%
df['Gpu brand'].value_counts()

# %%
sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()

# %%
df.drop(columns=['Gpu'],inplace=True)

# %%
df.head()

# %%
df['OpSys'].value_counts()

# %%
sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

# %%
df['os'] = df['OpSys'].apply(cat_os)

# %%
df.drop(columns=['OpSys'],inplace=True)

# %%
sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

# %%
sns.distplot(df['Weight'])

# %%
sns.scatterplot(x=df['Weight'],y=df['Price'])

# %%
sns.distplot(np.log(df['Price']))

# %%
X = df.drop(columns=['Price'])
y = np.log(df['Price'])

# %%
X

# %%
y

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

# %%
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# %% [markdown]
# ### Linear regression

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
], remainder='passthrough')

step2 = LinearRegression()

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Train model
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Evaluation
print('R2 score:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))

# %% [markdown]
# ### Ridge Regression

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### Lasso Regression

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)


pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### Decision Tree

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = DecisionTreeRegressor(max_depth=8)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### SVM

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = SVR(kernel='rbf',C=10000,epsilon=0.1)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### Random Forest

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### AdaBoost

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = AdaBoostRegressor(n_estimators=15,learning_rate=1.0)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### Gradient Boost

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = GradientBoostingRegressor(n_estimators=500)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### XgBoost

# %%
step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')

step2 = XGBRegressor(n_estimators=45,max_depth=5,learning_rate=0.5)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### Stacking

# %%
from sklearn.ensemble import StackingRegressor

step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
],remainder='passthrough')


estimators = [
    ('rf', RandomForestRegressor(n_estimators=350,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)),
    ('gbdt',GradientBoostingRegressor(n_estimators=100,max_features=0.5)),
    ('xgb', XGBRegressor(n_estimators=25,learning_rate=0.3,max_depth=5))
]

step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_squared_error(y_test,y_pred))

# %% [markdown]
# ### Exporting the Model

# %%
import pickle

pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))

# %%
df.head()

# %%



