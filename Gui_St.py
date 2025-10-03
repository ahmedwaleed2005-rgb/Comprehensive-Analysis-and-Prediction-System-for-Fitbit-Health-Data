#%%
import streamlit as st
from ml_functions import (
    predict_calories,predict_bmi,
    classify_bmi, classify_calories, classify_steps,
    classify_sleep, classify_sleep_efficiency,
    classify_activity_intensity, classify_hr
)
#%%
# ==============================
# واجهة Streamlit
# ==============================
st.set_page_config(page_title="Health AI Assistant", page_icon="💪", layout="centered")

st.title("💡 Health & Fitness AI Assistant")
st.write("اختر الخدمة اللي عايزها من الموديلات 👇")

# القائمة الرئيسية
service = st.selectbox(
    "اختار نوع الخدمة",
    [
        "🔮 Predict Calories (Regression)",
        "🔮 Predict BMI (Regression)",
        "📊 Classify BMI",
        "📊 Classify Calories",
        "📊 Classify Steps",
        "📊 Classify Sleep",
        "📊 Classify Sleep Efficiency",
        "📊 Classify Activity Intensity",
        "📊 Classify Heart Rate"
    ]
)
#%%
# =============================='TotalSteps', 'TotalDistance', 'TrackerDistance', 'LightActiveDistance',
    #    'LightlyActiveMinutes', 'SedentaryMinutes', 'IsManualReport',
    #    'TotalActiveMinutes', 'HR_mean', 'HR_std'
# إدخال البيانات + التنبؤ
# ==============================
if service == "🔮 Predict Calories (Regression)":
    st.subheader("تنبؤ بالسعرات الحرارية 🔥")

    TotalSteps          = st.number_input("عدد الخطوات", min_value=0)
    TotalDistance       = st.number_input("المسافة الكلية (كم)", min_value=0)
    TrackerDistance     = st.number_input("المسافة المسجلة بالجهاز (كم)", min_value=0)
    LightActiveDistance = st.number_input("المسافة في النشاط الخفيف (كم)", min_value=0)
    LightlyActiveMinutes = st.number_input("عدد الدقايق نشاط خفيف", min_value=0)
    SedentaryMinutes    = st.number_input("عدد الدقايق خمول", min_value=0)
    IsManualReport      = st.number_input("هل التقرير يدوي (0/1)", min_value=0, max_value=1)
    TotalActiveMinutes  = st.number_input("إجمالي دقائق النشاط", min_value=0)
    HR_mean             = st.number_input("متوسط معدل ضربات القلب", min_value=0)
    HR_std              = st.number_input("انحراف ضربات القلب", min_value=0)

    if st.button("احسب السعرات"):
        features = {
            "TotalSteps": TotalSteps,
            "TotalDistance": TotalDistance,
            "TrackerDistance": TrackerDistance,
            "LightActiveDistance": LightActiveDistance,
            "LightlyActiveMinutes": LightlyActiveMinutes,
            "SedentaryMinutes": SedentaryMinutes,
            "IsManualReport": IsManualReport,
            "TotalActiveMinutes": TotalActiveMinutes,
            "HR_mean": HR_mean,
            "HR_std": HR_std,
        }

        result = predict_calories(features)
        st.success(f"🔥 السعرات المتوقعة: {result:.2f} Cal")
 

elif service == "🔮 Predict BMI (Regression)":
    st.subheader("تنبؤ بـ BMI ⚖️")

    WeightKg       = st.number_input("الوزن (kg)", min_value=0)
    WeightPounds   = st.number_input("الطول (cm)", min_value=0)
    Fat            = st.number_input("نسبة الدهون", min_value=0)
    CaloriesPerStep = st.number_input("السعرات لكل خطوة", min_value=0)
    BMI_Category   = st.number_input("فئة الـ BMI", min_value=0)

    if st.button("احسب BMI"):
        BMI_features = {
            "WeightKg": WeightKg,
            "WeightPounds": WeightPounds,
            "Fat": Fat,
            "CaloriesPerStep": CaloriesPerStep,
            "BMI_Category": BMI_Category,
        }
        result = predict_bmi(list(BMI_features.values()))
        st.success(f"⚖️ BMI المتوقع: {result:.2f}")

#%%
# WeightKg	WeightPounds	Fat	BMI	CaloriesPerStep

if service == "📊 Classify BMI":

    st.subheader("تصنيف BMI Category")
    WeightKg       = st.number_input("الوزن (kg)", min_value=0)
    WeightPounds   = st.number_input("الطول (cm)", min_value=0)
    Fat            = st.number_input("نسبة الدهون", min_value=0)
    CaloriesPerStep = st.number_input("السعرات لكل خطوة", min_value=0)
    BMI   = st.number_input("BMI Value", min_value=0)

    if st.button("تحديد التصنيف"):
        BMI_features_Class = {
            "WeightKg": WeightKg,
            "WeightPounds": WeightPounds,
            "Fat": Fat,
            "CaloriesPerStep": CaloriesPerStep,
            "BMI_Category": BMI,
        }
        result = classify_bmi(list(BMI_features_Class.values()))
        st.success(f"⚖️ BMI التصنيف المتوقع: {result:.2f}")




elif service == "📊 Classify Calories":
    st.subheader("تصنيف السعرات")
    steps = st.number_input("عدد الخطوات", min_value=0)
    hr    = st.number_input("معدل ضربات القلب", min_value=0)
    if st.button("تحديد التصنيف"):
        result = classify_calories([steps, hr])
        st.info(f"📊 التصنيف المتوقع: {result}")

elif service == "📊 Classify Steps":
    st.subheader("تصنيف خطوات المستخدم")
    steps = st.number_input("عدد الخطوات", min_value=0)
    sleep = st.number_input("عدد ساعات النوم", min_value=0)
    if st.button("تحديد التصنيف"):
        result = classify_steps([steps, sleep])
        st.info(f"📊 التصنيف المتوقع: {result}")

elif service == "📊 Classify Sleep":
    st.subheader("تصنيف جودة النوم")
    sleep = st.number_input("عدد ساعات النوم", min_value=0)
    hr    = st.number_input("معدل ضربات القلب", min_value=0)
    if st.button("تحديد التصنيف"):
        result = classify_sleep([sleep, hr])
        st.info(f"😴 التصنيف المتوقع: {result}")

elif service == "📊 Classify Sleep Efficiency":
    st.subheader("تصنيف كفاءة النوم")
    sleep = st.number_input("عدد ساعات النوم", min_value=0)
    bed   = st.number_input("الوقت في السرير (دقائق)", min_value=0)
    if st.button("تحديد التصنيف"):
        result = classify_sleep_efficiency([sleep, bed])
        st.info(f"🛌 الكفاءة المتوقعة: {result}")

elif service == "📊 Classify Activity Intensity":
    st.subheader("تصنيف شدة النشاط")
    active_minutes = st.number_input("عدد الدقايق النشطة", min_value=0)
    steps = st.number_input("عدد الخطوات", min_value=0)
    if st.button("تحديد التصنيف"):
        result = classify_activity_intensity([active_minutes, steps])
        st.info(f"💪 الشدة المتوقعة: {result}")

elif service == "📊 Classify Heart Rate":
    st.subheader("تصنيف معدل ضربات القلب")
    hr_mean = st.number_input("متوسط معدل ضربات القلب", min_value=0)
    hr_std  = st.number_input("انحراف معدل ضربات القلب", min_value=0)
    if st.button("تحديد التصنيف"):
        result = classify_hr([hr_mean, hr_std])
        st.info(f"❤️ التصنيف المتوقع: {result}")
#%%

import tkinter as tk
from tkinter import messagebox
import joblib

# تحميل موديل مثال
model = joblib.load("models/BMI_Category_Random_Forest.pkl")

def predict():
    try:
        value = float(entry.get())
        # مثال: هنفترض الموديل بياخد قيمة واحدة
        pred = model.predict([[value]])
        messagebox.showinfo("Prediction", f"النتيجة: {pred[0]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Health Prediction System")

tk.Label(root, text="أدخل قيمة:").pack(pady=5)
entry = tk.Entry(root)
entry.pack(pady=5)

tk.Button(root, text="توقع", command=predict).pack(pady=10)

root.mainloop()
