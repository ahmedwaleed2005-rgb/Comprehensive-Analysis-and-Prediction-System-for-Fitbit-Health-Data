#%%
#========= Import Libraries ================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
#-- for regression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, median_absolute_error, r2_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVR
from sklearn.metrics import PredictionErrorDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#-- for classifaction
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectKBest, chi2      
import os
from sklearn.feature_selection import SelectKBest, f_regression,f_classif
import joblib
# print(len(10))
#%%
#========================================================
# Exploering Data
#========================================================
Daily_Activity_Data = pd.read_csv(r'F:\test\projects\NewProject\mturkfitbit_export_3.12.16-4.11.16\Fitabase Data 3.12.16-4.11.16\dailyActivity_merged.csv')
print(Daily_Activity_Data.describe())
#%%
Daily_Activity_Data.shape
print(Daily_Activity_Data.columns)
print(Daily_Activity_Data.info())
print(Daily_Activity_Data.isna().sum())
Daily_Activity_Data.head()
#%%
Sleep_Day_Data = pd.read_csv(r'F:\test\projects\NewProject\mturkfitbit_export_4.12.16-5.12.16\Fitabase Data 4.12.16-5.12.16\sleepDay_merged.csv')
# print(Sleep_Day_Data.describe())
# Sleep_Day_Data.shape


# print(Sleep_Day_Data.columns)
# print(Sleep_Day_Data.info())
# print(Sleep_Day_Data.isna().sum())
# Sleep_Day_Data.head()
#%%
Heartrate_Seconds_Data = pd.read_csv(r'F:\test\projects\NewProject\mturkfitbit_export_4.12.16-5.12.16\Fitabase Data 4.12.16-5.12.16\heartrate_seconds_merged.csv')
# print(Heartrate_Seconds_Data.d
# escribe())
# Heartrate_Seconds_Data.shape
# print(Heartrate_Seconds_Data.columns)
# print(Heartrate_Seconds_Data.info())
# print(Heartrate_Seconds_Data.isna().sum())
Heartrate_Seconds_Data.head()

#%%
Weight_LogInfo_Data = pd.read_csv(r'F:\test\projects\NewProject\mturkfitbit_export_4.12.16-5.12.16\Fitabase Data 4.12.16-5.12.16\weightLogInfo_merged.csv') 
(Weight_LogInfo_Data.describe())
Weight_LogInfo_Data.shape


def fill_fat(row):
    if pd.isna(row['Fat']):
        return 10 if row['BMI'] < 25 else 22
    return row['Fat']

Weight_LogInfo_Data['Fat'] = Weight_LogInfo_Data.apply(fill_fat, axis=1)


print(Weight_LogInfo_Data.columns)
print(Weight_LogInfo_Data.info())
print(Weight_LogInfo_Data.isna().sum())
Weight_LogInfo_Data.head()
#========================================================
# End of Exploering Data
#========================================================


#%%
#=================================================
#Start Treating With Data
#=================================================
Daily_Activity_Data['ActivityDate'] = pd.to_datetime(Daily_Activity_Data['ActivityDate']).dt.date

# نخلي SleepDay كمان تاريخ (من غير وقت)
Sleep_Day_Data['SleepDay'] = pd.to_datetime(Sleep_Day_Data['SleepDay']).dt.date

mergData = pd.merge(
    Daily_Activity_Data, 
    Sleep_Day_Data, 
    on= ['Id',], 
    how='inner'
)
#%%

hr_stats = Heartrate_Seconds_Data.groupby('Id')['Value'].agg(['mean','max','min','std']).reset_index()
hr_stats.rename(columns={
    'mean':'HR_mean', 'max':'HR_max', 'min':'HR_min', 'std':'HR_std'
}, inplace=True)

sampled_hr = Heartrate_Seconds_Data.sample(n=1000, random_state=42).reset_index(drop=True)
NewMergData = pd.merge(mergData , sampled_hr, on='Id', how='inner')
FinalData = pd.merge(NewMergData, Weight_LogInfo_Data, on=['Id'], how='inner')

FinalData.shape
#%%
FinalData.head()
FinalData.columns
FinalData.isna().sum()
#%%
#=================================================
# Feature Engineering
#=================================================

# Features من التواريخ
FinalData['DayOfWeek'] = pd.to_datetime(FinalData['ActivityDate']).dt.day_name()
FinalData['Month'] = pd.to_datetime(FinalData['ActivityDate']).dt.month
FinalData['IsWeekend'] = pd.to_datetime(FinalData['ActivityDate']).dt.dayofweek.isin([5, 6]).astype(int)

#%%
# Features من النشاط
FinalData['TotalActiveMinutes'] = (
    FinalData['VeryActiveMinutes'] + 
    FinalData['FairlyActiveMinutes'] + 
    FinalData['LightlyActiveMinutes']
)

FinalData['ActiveRatio'] = FinalData['TotalActiveMinutes'] / (
    FinalData['TotalActiveMinutes'] + FinalData['SedentaryMinutes'] + 1
)

FinalData['ActivityIntensity'] = FinalData['VeryActiveDistance'] / (FinalData['VeryActiveMinutes'] + 1)

FinalData['CaloriesPerStep'] = FinalData['Calories'] / (FinalData['TotalSteps'] + 1)

#%%
# Features من النوم
FinalData['SleepEfficiency'] = FinalData['TotalMinutesAsleep'] / (FinalData['TotalTimeInBed'] + 1)
FinalData['IsFragmentedSleep'] = (FinalData['TotalSleepRecords'] > 1).astype(int)

#%%
# Features من الوزن
def bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal'
    elif bmi < 30: return 'Overweight'
    else: return 'Obese'

FinalData['BMI_Category'] = FinalData['BMI'].apply(bmi_category)
#%%
#Interaction Features
FinalData['ActiveCaloriesRatio'] = FinalData['Calories'] / (FinalData['TotalActiveMinutes'] + 1)

# نفترض إن عمود Value هو الـ Heart Rate
FinalData['HR_mean'] = FinalData.groupby('Id')['Value'].transform('mean')
FinalData['HR_std'] = FinalData.groupby('Id')['Value'].transform('std')

# كفاءة النوم
FinalData['SleepEfficiency'] = FinalData['TotalMinutesAsleep'] / FinalData['TotalTimeInBed']

# مؤشر شدة النشاط
FinalData['ActivityIntensityIndex'] = (
    (FinalData['VeryActiveMinutes']*3 +
     FinalData['FairlyActiveMinutes']*2 +
     FinalData['LightlyActiveMinutes']*1) /
    (FinalData['SedentaryMinutes'] + 1) # +1 عشان نتفادى القسمة على صفر
)


#=================================================
#النتيجة
#=================================================
print(FinalData.shape)
print(FinalData.columns)
#%%
#===================================
# Data Visualzation
#===================================


# نضبط الشكل العام
plt.style.use("seaborn-v0_8")
sns.set_palette("viridis")

# دالة عامة للرسوم البيانية
def quick_plot(kind, data=None, x=None, y=None, title="", **kwargs):
    plt.figure(figsize=kwargs.pop("figsize", (8,5)))
    if kind == "hist":
        sns.histplot(data=data, x=x, kde=True, **kwargs)
    elif kind == "scatter":
        sns.scatterplot(data=data, x=x, y=y, **kwargs)
    elif kind == "box":
        sns.boxplot(data=data, x=x, y=y, **kwargs)
    elif kind == "line":
        sns.lineplot(data=data, x=x, y=y, **kwargs)
    elif kind == "kde":
        sns.kdeplot(data[x], shade=True, **kwargs)
    elif kind == "heatmap":
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", **kwargs)
    plt.title(title)
    plt.show()

# أمثلة سريعة من غير تكرار كتير:
quick_plot("hist", data=FinalData, x="TotalSteps", title="Distribution of Total Steps", bins=30)
quick_plot("hist", data=FinalData, x="SleepEfficiency", title="Distribution of Sleep Efficiency", bins=20)
quick_plot("scatter", data=FinalData, x="TotalSteps", y="Calories", title="Calories vs Total Steps")
quick_plot("scatter", data=FinalData, x="TotalActiveMinutes", y="TotalMinutesAsleep", title="Activity vs Sleep Duration")
quick_plot("hist", data=FinalData, x="HR_mean", title="Distribution of Mean Heart Rate", bins=25)
quick_plot("box", data=FinalData, x="BMI_Category", y="HR_mean", title="Heart Rate by BMI Category")
quick_plot("line", data=FinalData, x="ActivityDate", y="Calories", title="Calories Burned Over Time", figsize=(10,5))
quick_plot("line", data=FinalData, x="ActivityDate", y="TotalSteps", title="Steps Over Time", figsize=(10,5))
quick_plot("box", data=FinalData, x="DayOfWeek", y="SleepEfficiency", title="Sleep Efficiency by Day of Week")
quick_plot("scatter", data=FinalData, x="TotalActiveMinutes", y="SedentaryMinutes", title="Active vs Sedentary Minutes")
quick_plot("hist", data=FinalData, x="CaloriesPerStep", title="Distribution of Calories Per Step", bins=25)
quick_plot("scatter", data=FinalData, x="WeightKg", y="Calories", title="Weight vs Calories Burned")
quick_plot("box", data=FinalData, x="BMI_Category", y="TotalMinutesAsleep", title="Sleep Duration by BMI Category")
quick_plot("heatmap", data=FinalData, title="Correlation Heatmap of Features", figsize=(12,8))
#%%
def px_scatter(df, x, y, **kwargs):
    fig = px.scatter(df, x=x, y=y, **kwargs)
    fig.show()

def px_hist(df, x, **kwargs):
    fig = px.histogram(df, x=x, **kwargs)
    fig.show()

def px_sunburst(df, path, values, **kwargs):
    fig = px.sunburst(df, path=path, values=values, **kwargs)
    fig.show()

def px_scatter3d(df, x, y, z, **kwargs):
    fig = px.scatter_3d(df, x=x, y=y, z=z, **kwargs)
    fig.show()

# أمثلة
px_scatter(FinalData, x="TotalSteps", y="Calories", color="HR_mean",
           title="Interactive Steps vs Calories", hover_data=['BMI'])
px_hist(FinalData, x="TotalMinutesAsleep", nbins=40, color="BMI",
        title="Interactive Sleep Duration Distribution")
px_sunburst(FinalData, path=['Id','ActivityDate'], values='TotalSteps',
            title="Activity Breakdown by User and Date")
px_scatter3d(FinalData, x='TotalSteps', y='Calories', z='BMI',
             color='HR_mean', title="3D Relationship: Steps, Calories, BMI")


#==========================================
#------------------------------------------
#==========================================
#------------------------------------------
#------------------------------------------

# %% )=>Buliding_Model
#1=> Determine What is will be a target
x1 = FinalData.drop(['Id' ,'Calories'], axis=1)
y1 = FinalData['Calories']

x1.select_dtypes(include='object')
object_columns = ['ActivityDate','SleepDay', 'Date', 'Time', 'DayOfWeek','BMI_Category']

for ObjColm in object_columns:
    le =LabelEncoder()
    x1[ObjColm] = le.fit_transform(x1[ObjColm].astype(str))


#--feture Selecation
Selector = SelectKBest(score_func=f_regression, k=10)
x_Selected = Selector.fit_transform(x1,y1)
selected_features = x1.columns[Selector.get_support()]
print("Selected Features:", selected_features)
x_Selected= pd.DataFrame(x_Selected, columns=selected_features)



#%%
x1_train,x1_test,y1_train,y1_test = train_test_split(x_Selected ,y1,
                                                 test_size=0.2,
                                                 random_state=45,
                                                 shuffle=True,
                                               )
#%%
#=> هنا احنا بناخد عينات من x_train علشان نعرف ندربها في مودل الريجريشن اللي تحت لان حجمها كبير 
x1_train_small = x1_train.sample(100000, random_state=33)
#=> هنا احنا بناخد عينات من y_train وبناخدها بعدد يساوي اللي اتاخد من الx_train
y1_train_small = y1_train.loc[x1_train_small.index]

#%%
Models = {
    "LinearRegression": LinearRegression(fit_intercept=True, n_jobs=-1),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        loss='absolute_error', random_state=45, n_estimators=100, max_depth=6, max_features=20
    ),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=100, n_jobs=-1, max_depth=5, max_samples=20
    ),
    "LinearSVR": LinearSVR(
        random_state=45, loss='squared_epsilon_insensitive',
        intercept_scaling=2, fit_intercept=True
    ),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1, p=5),
    "DecisionTreeRegressor": DecisionTreeRegressor(
        random_state=45, max_depth=45, splitter='best', criterion='squared_error'
    ),
    "SGDRegressor": SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000),
    "Lasso": Lasso(alpha=1.0, random_state=33),
    "Ridge": Ridge(alpha=1, random_state=45, max_iter=1000),
}

#%%
Results = []

for name,Model in Models.items():
    # name = str(Model).split("(")[0]
    print(f"Training {name}...")

    # Train
    Model.fit(x1_train_small, y1_train_small)

    # Predict
    y_pred = Model.predict(x1_test)

    # Metrics
    mae = mean_absolute_error(y1_test, y_pred)
    mse = mean_squared_error(y1_test, y_pred)
    rmse = np.sqrt(mse)
    mdse = median_absolute_error(y1_test, y_pred)
    r2 = r2_score(y1_test, y_pred)
    mape = np.mean(np.abs((y1_test - y_pred) / y1_test)) * 100

    # Save results
    Results.append({
        "Model": name,
        "Train Score": Model.score(x1_train_small, y1_train_small),
        "Test Score": Model.score(x1_test, y1_test),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MdSE": mdse,
        "R2": r2,
        "MAPE": mape
    })
#%%
# Convert to DataFrame
results_df_Reg = pd.DataFrame(Results)

# Sort by best R2 (or lowest RMSE if تحب)
results_df_Reg = results_df_Reg.sort_values(by="R2", ascending=False)
print("\n Evaluation Results:\n", results_df_Reg)

# أفضل موديل
# أفضل موديل
best_model_name1 = results_df_Reg.iloc[0]["Model"]
print(f"\n Best Model: {best_model_name1}")

# رجّع الموديل نفسه من اللستة Models
# best_model = None

# for Model in Models:
#     name = str(Model).split("(")[0]
#     if name == best_model_name1:
#         best_model = Model
#         break

# اعمل prediction تاني من أفضل موديل
y_pred_best = Models[best_model_name1].predict(x1_test)
#%%
os.makedirs("models", exist_ok=True)

reg_final_model1 = Models[best_model_name1]

# اسم الملف
model_filename = f"models/{'BMI'}_{best_model_name1.replace(' ', '_')}.pkl"

# حفظ الموديل
joblib.dump(Models[best_model_name1], model_filename)
print(f"💾 Model saved: {model_filename}")

#%%
# print(results_df_Reg)
# print(results_df_Reg["Model"].unique())

#%%
# رسم كفاءة الموديلات
plt.figure(figsize=(12,6))
sns.barplot(data=results_df_Reg.reset_index(drop=True), x="Model", y="R2")
plt.title("Model Performance (R² Score)")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(data=results_df_Reg.reset_index(drop=True), x="Model", y="RMSE", palette="mako")
plt.title("Model Performance (RMSE)")
plt.ylabel("RMSE (lower is better)")
plt.xticks(rotation=45)
plt.show()

#%%
# residuals = y1_test - y_pred_best
# plt.figure(figsize=(8,5))
# sns.histplot(residuals, bins=30, kde=True, color="blue")
# plt.title("Distribution of Residuals")
# plt.show()
#%%
plt.figure(figsize=(10,6))
sns.heatmap(results_df_Reg.set_index("Model")[["MAE","RMSE","R2"]], 
            annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Error Metrics Heatmap")
plt.show()

#%%
plt.figure(figsize=(8,6))
plt.scatter(y1_test, y_pred, alpha=0.5)
plt.plot([y1_test.min(), y1_test.max()],
         [y1_test.min(), y1_test.max()],
         'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Best Model)")
plt.show()

#==========================================================================
#%%========================================================================
#==========================================================================
# Bulding_Models_That_predict BMI Value
#1=> Determine What is will be a target
x2 = FinalData.drop(['Id' ,'BMI'], axis=1)
y2 = FinalData['BMI']

x2.select_dtypes(include='object')
object_columns = ['ActivityDate','SleepDay', 'Date', 'Time', 'DayOfWeek','BMI_Category']

for ObjColm in object_columns:
    le =LabelEncoder()
    x2[ObjColm] = le.fit_transform(x2[ObjColm].astype(str))

#--feture Selecation
Selector2 = SelectKBest(score_func=f_regression, k=5)
x_Selected2 = Selector2.fit_transform(x2,y2)
selected_features2 = x2.columns[Selector2.get_support()]
print("Selected Features:", selected_features2)
x_Selected2= pd.DataFrame(x_Selected2, columns=selected_features2)
#%%
x2_train,x2_test,y2_train,y2_test = train_test_split(x_Selected2,y2,
                                                 test_size=0.2,
                                                 random_state=45,
                                                 shuffle=True,
                                                 )
#%%
x2_train_Samble = x2_train.sample(100000, random_state=45)
y2_train_Samble = y2_train.loc[x2_train_Samble.index]

#%%
# Bulding_Models_That_predict BMI Value
# LinearRegressionModel = LinearRegression(fit_intercept= True, n_jobs=-1, )
# GBRModel = GradientBoostingRegressor(loss='absolute_error',random_state=45,n_estimators=100,max_depth=6,max_features=20)
# RandomForestRegressorModel = RandomForestRegressor(n_estimators=100, n_jobs=-1,max_depth=5,max_samples=20)
# SVRModel = LinearSVR(random_state= 45, loss='squared_epsilon_insensitive' ,intercept_scaling=2,fit_intercept=True)
# KNeighborsRegressorModel = KNeighborsRegressor(n_neighbors=5,n_jobs=-1,p=5)
# DecisionTreeRegressorModel =DecisionTreeRegressor(random_state=45,max_depth=45,splitter='best',criterion='squared_error')
# SGDRegressionModel = SGDRegressor(loss='squared_error' ,penalty='l2',max_iter=1000)
# LassoRegressionModel = Lasso(alpha=1.0, random_state=33)
# RidgeRegressionModel =Ridge(alpha=1, random_state=45,max_iter=1000)

#=> List of All Models (Linear + Complex)
AllModels = {
    "LinearRegression": LinearRegression(fit_intercept=True, n_jobs=-1),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        loss='absolute_error', random_state=45, n_estimators=100, max_depth=6, max_features=20
    ),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=100, n_jobs=-1, max_depth=5, max_samples=20
    ),
    "LinearSVR": LinearSVR(
        random_state=45, loss='squared_epsilon_insensitive',
        intercept_scaling=2, fit_intercept=True
    ),
    "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=5, n_jobs=-1, p=5),
    "DecisionTreeRegressor": DecisionTreeRegressor(
        random_state=45, max_depth=45, splitter='best', criterion='squared_error'
    ),
    "SGDRegressor": SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000),
    "Lasso": Lasso(alpha=1.0, random_state=33),
    "Ridge": Ridge(alpha=1, random_state=45, max_iter=1000),
}


#%%=> Loop for All Models
Final_Result_All = []
for modelname, model in AllModels.items():
    # modelname = str(model).split("(")[0]
    print(f" Training: {modelname} ...")
    model.fit(x2_train_Samble, y2_train_Samble)
    print(f" Done: {modelname}")


    y2_pred = model.predict(x2_test)

    # Metrics
    MSE = mean_squared_error(y2_test, y2_pred)
    MAE = mean_absolute_error(y2_test, y2_pred)
    Median_AE = median_absolute_error(y2_test, y2_pred)
    R2_Score = r2_score(y2_test, y2_pred)

    Final_Result_All.append({
        'Model Name': modelname,
        'Train Score': model.score(x2_train_Samble, y2_train_Samble),
        'Test Score': model.score(x2_test, y2_test),
        'MAE': MAE,
        'MSE': MSE,
        'Median_AE': Median_AE,
        'R2_Score': R2_Score
    })
    
Final_Result_df = pd.DataFrame(Final_Result_All)
Final_Result_df = Final_Result_df.sort_values(by='R2_Score', ascending=False)

print("\n Evaluation Results:\n", Final_Result_df)

best_model_name2 = Final_Result_df.iloc[0]['Model Name']
print(f"🔥 The Best Model is {best_model_name2}")
os.makedirs("models", exist_ok=True)

reg_final_model = AllModels[best_model_name2]

# اسم الملف
model_filename = f"models/{'BMI'}_{best_model_name2.replace(' ', '_')}.pkl"

# حفظ الموديل
joblib.dump(AllModels[best_model_name2], model_filename)
print(f"💾 Model saved: {model_filename}")

#%% Visualization for All Models

# 1️⃣ Barplot - R² Score
plt.figure(figsize=(12,6))
sns.barplot(data=Final_Result_df, x='Model Name', y='R2_Score', palette="viridis")
plt.title("Model Performance (R² Score)")
plt.ylabel('R2_Score')
plt.xticks(rotation=45)
plt.show()

# 2️⃣ Barplot - MAE
plt.figure(figsize=(12,6))
sns.barplot(data=Final_Result_df, x='Model Name', y='MAE', palette="mako")
plt.title("Model Performance (MAE)")
plt.ylabel("MAE (lower is better)")
plt.xticks(rotation=45)
plt.show()

# 3️⃣ Residuals Distribution (Best Model)
# best_model_row = Final_Result_df.iloc[0]
# best_model_name2 = best_model_row['Model Name']

# # إعادة تدريب أفضل موديل على الداتا
# best_model = [m for m in AllModels if str(m).split("(")[0] == best_model_name2][0]
# best_model.fit(x2_train_Samble, y2_train_Samble)
# y2_pred = best_model.predict(x2_test)

best_model_name = Final_Result_df.iloc[0]['Model Name']

# الموديل نفسه من الديكشنري
best_model = AllModels[best_model_name]

# إعادة تدريب أفضل موديل على الداتا
best_model.fit(x2_train_Samble, y2_train_Samble)
y2_pred = best_model.predict(x2_test)

residuals = y2_test - y2_pred
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True, color="blue")
plt.title(f"Distribution of Residuals ({best_model_name2})")
plt.show()

# 4️⃣ Heatmap for All Error Metrics
plt.figure(figsize=(10,6))
sns.heatmap(Final_Result_df.set_index('Model Name')[['MSE','Median_AE','R2_Score']], 
            annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Error Metrics Heatmap (All Models)")
plt.show()

# 5️⃣ Actual vs Predicted (Best Model)
plt.figure(figsize=(8,6))
plt.scatter(y2_test, y2_pred, alpha=0.5)
plt.plot([y2_test.min(), y2_test.max()],
         [y2_test.min(), y2_test.max()],
         'r--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted ({best_model_name2})")
plt.show()


#%%==========================================================
#Now Bulding Classifaction Model
#===============================================================
FinalData.select_dtypes(include='object')
object_columns = ['ActivityDate','SleepDay', 'Date', 'Time', 'DayOfWeek','BMI_Category']

for ObjColm in object_columns:
    le =LabelEncoder()
    FinalData[ObjColm] = le.fit_transform(FinalData[ObjColm].astype(str))

Xclssifiaier1 =FinalData.drop(['Id' ,'BMI_Category'], axis=1)
yClassifaier1 = FinalData['BMI_Category']

#--feture Selecation
Selector_class1 = SelectKBest(score_func=f_classif, k=5)
x_Selected_class1 = Selector_class1.fit_transform(Xclssifiaier1,yClassifaier1)
selected_features_class1 = Xclssifiaier1.columns[Selector_class1.get_support()]
print("Selected Features:", selected_features_class1)
x_Selected_class1= pd.DataFrame(x_Selected_class1, columns=selected_features_class1)

#%%
XtrainClass1,XtestClass1,YtrainClass1,YtestClass1 = train_test_split(x_Selected_class1,yClassifaier1,
                                                                     random_state=45,
                                                                     shuffle=True,
                                                                     test_size=0.25)

#%%
#  -------------------------------------------------
# Train & Evaluate Models
# -------------------------------------------------
modelsclass1 = {
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100,
                                            max_depth=10, max_features=0.5, max_leaf_nodes=50), 
    'Logistic Regression': LogisticRegression(max_iter=1000,n_jobs=-1,random_state=45,penalty="l2"),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Linear SVC': LinearSVC(penalty="l2", loss="squared_hinge", multi_class="ovr",
                             random_state=45, max_iter=1000),
    'GaussianNB': GaussianNB(),
    'KNN': KNeighborsClassifier(n_neighbors=5,n_jobs=-1)
 }

results_Class1 = []

for nameclass1, modelclass1 in modelsclass1.items():
    try:
        modelclass1.fit(XtrainClass1, YtrainClass1)
        y_predClass1 = modelclass1.predict(XtestClass1)

        results_Class1.append({
            'Model class1': nameclass1,
            'Accuracy1': round(accuracy_score(YtestClass1,y_predClass1), 3),
            'Precision1': round(precision_score(YtestClass1,y_predClass1, average='weighted'), 3),
            'Recall1': round(recall_score(YtestClass1,y_predClass1, average='weighted'), 3),
            'F1 Score class1': round(f1_score(YtestClass1,y_predClass1, average='weighted'), 3)
        })
    except Exception as e:
        print(f"[!] Model {nameclass1} failed: {e}")
#%% -------------------------------------------------
# Results Comparison
# -------------------------------------------------
results_Class_df1 = pd.DataFrame(results_Class1).sort_values(by='F1 Score class1', ascending=False).reset_index(drop=True)
print("\n Model Comparison:\n")
print(results_Class_df1)

best_model_name_class1 = results_Class_df1.iloc[0]['Model class1']
print(f"Best model: {best_model_name_class1}")

final_model_class1 = modelsclass1[best_model_name_class1]
final_model_class1.fit(XtrainClass1, YtrainClass1)
y_test_pred = final_model_class1.predict(XtestClass1)

os.makedirs("models", exist_ok=True)

BMI_CLASS_FINALL_MODEL = modelsclass1[best_model_name_class1]

# اسم الملف
BMIClass_model_filename = f"models/{'BMI'}_{best_model_name_class1.replace(' ', '_')}.pkl"

# حفظ الموديل
joblib.dump(AllModels[best_model_name2], BMIClass_model_filename)
print(f"💾 Model saved: {BMIClass_model_filename}")


cm = confusion_matrix(YtestClass1, y_test_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()
#%% -------------------------------------------------
# Visualization of Results
# -------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=results_Class_df1, x='Model class1', y='F1 Score class1', palette='viridis')
plt.xticks(rotation=30, ha='right')
plt.title('F1 Score Comparison Between Models', fontsize=16)
plt.show()

# Feature Importance (RandomForest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(Xclssifiaier1, yClassifaier1)
feat_importance = pd.Series(rf_model.feature_importances_, index=Xclssifiaier1.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.title('Top 8 Feature Importances')
plt.show()
#%%
#==========================================================
# Create Categorical Columns (Targets for Classification)
#==========================================================

#%%

# Calories -> 3 فئات
FinalData['Calories_Class'] = pd.cut(
    FinalData['Calories'],
    bins=[-1, 1800, 3000, FinalData['Calories'].max()],
    labels=['Low', 'Medium', 'High']
)

# Steps -> 4 فئات
FinalData['Steps_Class'] = pd.cut(
    FinalData['TotalSteps'],
    bins=[-1, 5000, 10000, 15000, FinalData['TotalSteps'].max()],
    labels=['Sedentary', 'Lightly Active', 'Active', 'Highly Active']
)

# Sleep Duration -> 3 فئات
FinalData['Sleep_Class'] = pd.cut(
    FinalData['TotalMinutesAsleep'],
    bins=[-1, 300, 480, FinalData['TotalMinutesAsleep'].max()],  
    labels=['Short', 'Normal', 'Long']
)

# Sleep Efficiency = (Minutes Asleep / Minutes in Bed) * 100 -> 3 فئات
FinalData['SleepEfficiency'] = (FinalData['TotalMinutesAsleep'] / FinalData['TotalTimeInBed']) * 100
FinalData['SleepEfficiency_Class'] = pd.cut(
    FinalData['SleepEfficiency'],
    bins=[-1, 75, 90, 100],
    labels=['Low Efficiency', 'Medium Efficiency', 'High Efficiency']
)

# Activity Intensity (Minutes Very Active) -> 3 فئات
FinalData['ActivityIntensity_Class'] = pd.cut(
    FinalData['VeryActiveMinutes'],
    bins=[-1, 30, 60, FinalData['VeryActiveMinutes'].max()],
    labels=['Low Intensity', 'Moderate Intensity', 'High Intensity']
)

# Heart Rate -> 3 فئات
# نجيب الماكس الحقيقي
hr_max = FinalData['HR_mean'].max()

# لو الماكس أقل من 90 هنظبط الـ bins بحيث تفضل مرتبة
if hr_max <= 90:
    bins = [-1, 60, hr_max]  # يعني Low / Normal
    labels = ['Low HR', 'Normal HR']
else:
    bins = [-1, 60, 90, hr_max]
    labels = ['Low HR', 'Normal HR', 'High HR']

FinalData['HR_Class'] = pd.cut(FinalData['HR_mean'], bins=bins, labels=labels)


# Check quickly
FinalData[['Calories_Class','Steps_Class','Sleep_Class',
           'SleepEfficiency_Class','ActivityIntensity_Class','HR_Class']].head()

FinalData_sample = FinalData.sample(20000, random_state=42)  # أو أي رقم مناسب
FinalData_sample.select_dtypes(include='object')
object_columns =['ActivityDate', 'SleepDay', 'Date',
                  'Time', 'DayOfWeek', 'BMI_Category',
                    'Calories_Class', 'Steps_Class', 'Sleep_Class',
                      'SleepEfficiency_Class', 'ActivityIntensity_Class'
                      , 'Steps_Class', 'Sleep_Class', 'SleepEfficiency_Class'
                      , 'ActivityIntensity_Class','HR_Class']
#%%HR_ClassCalories_Class
#%%==========================================================
# Function to Train & Evaluate Classification Models
# ============================================================
FinalData_sample.select_dtypes(include='object')
object_columns = ['ActivityDate','SleepDay', 'Date',
                    'Time', 'DayOfWeek','BMI_Category',
                    'Calories_Class','Steps_Class','Sleep_Class',
                    'SleepEfficiency_Class','ActivityIntensity_Class','HR_Class'
                    'Calories_Class','Steps_Class','Sleep_Class',
                    'SleepEfficiency_Class','ActivityIntensity_Class','HR_Class']
#%%
Target_classifaction_col = [
    "BMI_Category",          # موجود جاهز
    "IsManualReport",        # 0/1
    "IsWeekend",             # 0/1
    "IsFragmentedSleep",     # 0/1
    # اللي ممكن نعمله Binning ونضيفه كأعمدة جديدة
    "Calories_Class",        
    "Steps_Class",
    "Sleep_Class",
    "SleepEfficiency_Class",
    "ActivityIntensity_Class",
    "HR_Class"
]

for ObjColm in object_columns:
        le =LabelEncoder()
        FinalData_sample[ObjColm] = le.fit_transform(FinalData_sample[ObjColm].astype(str))

for target_col in Target_classifaction_col:

    print(f"\n{'='*60}")
    print(f"🎯 Target: {target_col}")
    print(f"{'='*60}\n")
    # التعامل مع الـ Features
    Xclssifiaier2 = FinalData_sample.drop(['Id', target_col], axis=1)
    yClassifaier2 = FinalData_sample[target_col]

    Selector_class2 = SelectKBest(score_func=f_regression, k=10)
    x_Selected_class2 = Selector_class2.fit_transform(Xclssifiaier2, yClassifaier2)
    selected_features_class2 = Xclssifiaier2.columns[Selector_class2.get_support()]
    print("Selected Features:", selected_features_class2)
    x_Selected_class2 = pd.DataFrame(x_Selected_class2, columns=selected_features_class2)


    # Train-Test Split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
        x_Selected_class2, yClassifaier2, random_state=45, shuffle=True, test_size=0.25
    )

    # Models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100,
                                                max_depth=10, max_features=0.5, max_leaf_nodes=50), 
        # 'Logistic Regression': LogisticRegression(max_iter=1000,n_jobs=-1,random_state=45,penalty="l2"),
        # 'Decision Tree': DecisionTreeClassifier(random_state=42),
        # 'GaussianNB': GaussianNB(),
    }

    # Results
    results = []
    for name, model in models.items():
        try:
            model.fit(Xtrain, Ytrain)
            y_pred = model.predict(Xtest)

            results.append({
                'Model': name,
                'Accuracy': round(accuracy_score(Ytest, y_pred), 3),
                'Precision': round(precision_score(Ytest, y_pred, average='weighted'), 3),
                'Recall': round(recall_score(Ytest, y_pred, average='weighted'), 3),
                'F1 Score': round(f1_score(Ytest, y_pred, average='weighted'), 3)
            })
        except Exception as e:
            print(f"[!] Model {name} failed: {e}")
    #%%
    # Results DataFrame
    results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
    print("\n Model Comparison:\n", results_df)
    #%%
    # Best Model
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n✅ Best model for {target_col}: {best_model_name}\n")

    final_model = models[best_model_name]
    final_model.fit( x_Selected_class2, yClassifaier2)

    # إنشاء فولدر models لو مش موجود
    os.makedirs("models", exist_ok=True)

    # اسم الملف
    model_filename = f"models/{target_col}_{best_model_name.replace(' ', '_')}.pkl"

    # حفظ الموديل
    joblib.dump(final_model, model_filename)
    print(f"💾 Model saved: {model_filename}")

    # Confusion Matrix
    y_test_pred = final_model.predict(Xtest)
    cm = confusion_matrix(Ytest, y_test_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title(f"Confusion Matrix - {target_col}")
    plt.show()

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Model', y='F1 Score', palette='viridis')
    plt.xticks(rotation=30, ha='right')
    plt.title(f'F1 Score Comparison for {target_col}', fontsize=16)
    plt.show()

    # Feature Importance (لو الموديل Random Forest)
    if best_model_name == 'Random Forest':
        rf_model = final_model
        feat_importance = pd.Series(rf_model.feature_importances_, index= x_Selected_class2.columns)
        feat_importance.nlargest(10).plot(kind='barh')
        plt.title(f'Top 10 Feature Importances - {target_col}')
        plt.show()

        
# #%%

# # أخذ عينة من البيانات
# FinalData_sample = FinalData.sample(20000, random_state=42)

# # الأعمدة الـ Object
# object_columns = [
#     'ActivityDate', 'SleepDay', 'Date', 'Time', 'DayOfWeek',
#     'BMI_Category', 'Calories_Class', 'Steps_Class', 'Sleep_Class',
#     'SleepEfficiency_Class', 'ActivityIntensity_Class', 'HR_Class'
# ]

# # الأعمدة اللي هنعمل عليها التصنيف
# Target_classifaction_col = [
#     "BMI_Category",          # موجود جاهز
#     "IsManualReport",        # 0/1
#     "IsWeekend",             # 0/1
#     "IsFragmentedSleep",     # 0/1
#     "Calories_Class",        
#     "Steps_Class",
#     "Sleep_Class",
#     "SleepEfficiency_Class",
#     "ActivityIntensity_Class",
#     "HR_Class"
# ]

# # تحويل الأعمدة الـ Object لأرقام
# for ObjColm in object_columns:
#     if ObjColm in FinalData_sample.columns:   # ✅ نتأكد إن العمود موجود
#         le = LabelEncoder()
#         FinalData_sample[ObjColm] = le.fit_transform(FinalData_sample[ObjColm].astype(str))

# # Loop على الأعمدة الهدف
# for target_col in Target_classifaction_col:

#     if target_col not in FinalData_sample.columns:   # ✅ لو العمود مش موجود نعمل Skip
#         print(f"[!] Skipping {target_col} (not found in FinalData_sample)")
#         continue

#     print(f"\n{'='*60}")
#     print(f"🎯 Target: {target_col}")
#     print(f"{'='*60}\n")

#     # التعامل مع الـ Features
#     X = FinalData_sample.drop(['Id', target_col], axis=1)
#     y = FinalData_sample[target_col]

#     # Train-Test Split
#     Xtrain, Xtest, Ytrain, Ytest = train_test_split(
#         X, y, random_state=45, shuffle=True, test_size=0.25
#     )

#     # Models
#     models = {
#         'Random Forest': RandomForestClassifier(
#             random_state=42, n_estimators=100,
#             max_depth=10, max_features=0.5, max_leaf_nodes=50
#         ),
#         # ممكن تفعل باقي الموديلات بعدين
#         # 'Logistic Regression': LogisticRegression(max_iter=1000,n_jobs=-1,random_state=45,penalty="l2"),
#         # 'Decision Tree': DecisionTreeClassifier(random_state=42),
#         # 'GaussianNB': GaussianNB(),
#     }

#     # Results
#     results = []
#     for name, model in models.items():
#         try:
#             model.fit(Xtrain, Ytrain)
#             y_pred = model.predict(Xtest)

#             results.append({
#                 'Model': name,
#                 'Accuracy': round(accuracy_score(Ytest, y_pred), 3),
#                 'Precision': round(precision_score(Ytest, y_pred, average='weighted'), 3),
#                 'Recall': round(recall_score(Ytest, y_pred, average='weighted'), 3),
#                 'F1 Score': round(f1_score(Ytest, y_pred, average='weighted'), 3)
#             })
#         except Exception as e:
#             print(f"[!] Model {name} failed: {e}")

#     # Results DataFrame
#     results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
#     print("\n Model Comparison:\n", results_df)

#     # Best Model
#     best_model_name = results_df.iloc[0]['Model']
#     print(f"\n✅ Best model for {target_col}: {best_model_name}\n")

#     final_model = models[best_model_name]
#     final_model.fit(X, y)

#     # إنشاء فولدر models لو مش موجود
#     os.makedirs("models", exist_ok=True)

#     # اسم الملف
#     model_filename = f"models/{target_col}_{best_model_name.replace(' ', '_')}.pkl"

#     # حفظ الموديل
#     joblib.dump(final_model, model_filename)
#     print(f"💾 Model saved: {model_filename}")

#     # Confusion Matrix
#     y_test_pred = final_model.predict(Xtest)
#     cm = confusion_matrix(Ytest, y_test_pred)
#     ConfusionMatrixDisplay(confusion_matrix=cm).plot()
#     plt.title(f"Confusion Matrix - {target_col}")
#     plt.show()

#     # Visualization
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=results_df, x='Model', y='F1 Score', palette='viridis')
#     plt.xticks(rotation=30, ha='right')
#     plt.title(f'F1 Score Comparison for {target_col}', fontsize=16)
#     plt.show()

#     # Feature Importance (لو الموديل Random Forest)
#     if best_model_name == 'Random Forest':
#         rf_model = final_model
#         feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
#         feat_importance.nlargest(10).plot(kind='barh')
#         plt.title(f'Top 10 Feature Importances - {target_col}')
#         plt.show()

