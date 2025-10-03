#%%
import joblib
import numpy as np
import pandas as pd

#%%
# ==============================
# تحميل الموديلات (Regression)
# ==============================
calories_reg_model = joblib.load("models/BMI_KNeighborsRegressor.pkl")
bmi_reg_model      = joblib.load("models/BMI_KNeighborsRegressor.pkl")
#%%
# ==============================
# تحميل الموديلات (Classification)
# ==============================
bmi_cls_model              = joblib.load("models/BMI_Category_Random_Forest.pkl")
calories_cls_model         = joblib.load("models/IsManualReport_Random_Forest.pkl")
steps_cls_model            = joblib.load("models/IsWeekend_Random_Forest.pkl")
sleep_cls_model            = joblib.load("models/IsFragmentedSleep_Random_Forest.pkl")
sleep_eff_cls_model        = joblib.load("models/Calories_Class_Random_Forest.pkl")
activity_intensity_cls_model = joblib.load("models/Steps_Class_Random_Forest.pkl")
hr_cls_model               = joblib.load("models/Sleep_Class_Random_Forest.pkl")
#%%
# ==============================
# Regression Functions
# # # ==============================0])

# # كل الفيتشرز اللي اتدرب عليها الموديل

all_features = calories_reg_model.feature_names_in_

# الفيتشرز اللي انت اخترتها
Caloris_selected_features = ['TotalSteps', 'TotalDistance', 'TrackerDistance', 'LightActiveDistance',
       'LightlyActiveMinutes', 'SedentaryMinutes', 'IsManualReport',
       'TotalActiveMinutes', 'HR_mean', 'HR_std']

def predict_calories(features: dict):
    """ توقع السعرات الحرارية """
    # نجهز DataFrame فيه كل الأعمدة
    input_data = pd.DataFrame([ {f:0 for f in all_features} ])  
    
    # نملأ الأعمدة المختارة بالقيم اللي المستخدم دخلها
    for f in Caloris_selected_features:
        input_data[f] = features.get(f, 0)

    # نعمل prediction
    return float(calories_reg_model.predict(input_data)[0])

#%%
all_BMI_features = bmi_reg_model.feature_names_in_
BMI_selected_features = ['WeightKg','WeightPounds',	'Fat','CaloriesPerStep','BMI_Category']
def predict_bmi(features: list):
    """ توقع BMI """
    # features = np.array(features).reshape(1, -1)
    # return float(bmi_reg_model.predict(features)[0])
    input_data_BMI = pd.DataFrame([ {f:0 for f in all_BMI_features} ])  
    
    # نملأ الأعمدة المختارة بالقيم اللي المستخدم دخلها
    for f_BMI in BMI_selected_features:
        input_data_BMI[f_BMI] = features.get(f_BMI, 0)

    # نعمل prediction
    return float(bmi_reg_model.predict(input_data_BMI)[0])
#%%
# ==============================
# Classification Functions
# ==============================


all_BMI_features_Class = bmi_cls_model.feature_names_in_
BMI_selected_features_class = ['WeightKg','WeightPounds','Fat','CaloriesPerStep','BMI']
def classify_bmi(features: list):
    """ توقع BMI """
    # features = np.array(features).reshape(1, -1)
    # return float(bmi_reg_model.predict(features)[0])
    input_data_BMI_Class = pd.DataFrame([ {f:0 for f in all_BMI_features_Class} ])  
    
    # نملأ الأعمدة المختارة بالقيم اللي المستخدم دخلها
    for f_BMI_class in BMI_selected_features_class:
        input_data_BMI_Class[f_BMI_class] = features.get(f_BMI_class, 0)

    # نعمل prediction
    return float(bmi_cls_model.predict(input_data_BMI_Class)[0])


# def classify_bmi(features: list):
#     features = np.array(features).reshape(1, -1)
#     return bmi_cls_model.predict(features)[0]

def classify_calories(features: list):
    features = np.array(features).reshape(1, -1)
    return calories_cls_model.predict(features)[0]

def classify_steps(features: list):
    features = np.array(features).reshape(1, -1)
    return steps_cls_model.predict(features)[0]

def classify_sleep(features: list):
    features = np.array(features).reshape(1, -1)
    return sleep_cls_model.predict(features)[0]

def classify_sleep_efficiency(features: list):
    features = np.array(features).reshape(1, -1)
    return sleep_eff_cls_model.predict(features)[0]

def classify_activity_intensity(features: list):
    features = np.array(features).reshape(1, -1)
    return activity_intensity_cls_model.predict(features)[0]

def classify_hr(features: list):
    features = np.array(features).reshape(1, -1)
    return hr_cls_model.predict(features)[0]
