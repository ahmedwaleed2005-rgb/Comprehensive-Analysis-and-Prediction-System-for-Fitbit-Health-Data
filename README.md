# Project Title
Comprehensive Analysis and Prediction System for Fitbit Health Data

# this is my project on kaggel you should open this link for find project data 
https://www.kaggle.com/code/ahmedwaleedsaleh/human-health-ml-model

# 📊 Fitbit Health Data Analysis & Prediction System

# 🚀 Executive Summary

This project delivers an end-to-end machine learning pipeline on Fitbit data 🏃‍♂️💤❤️⚖️, covering activity, sleep, heart rate, and weight. It extracts meaningful health insights, engineers features, and builds both regression 🔢 and classification 🧩 models for calorie prediction, BMI estimation, and lifestyle pattern classification.

# 🎯 Objectives

🔎 Generate descriptive insights from user activity, sleep, HR, and weight.

📈 Build regression models for Calories & BMI prediction.

🏷️ Train classification models for BMI Category, Activity, Sleep, and HR groups.

📊 Provide visual insights & dashboards.

💾 Save production-ready models for deployment.

# 📂 Data Sources

dailyActivity_merged.csv 🏃‍♀️

sleepDay_merged.csv 😴

heartrate_seconds_merged.csv ❤️

weightLogInfo_merged.csv ⚖️

Features: steps, activity minutes, calories, distances, sleep duration/efficiency, HR stats, weight, and BMI.

# 🛠️ Data Preparation

🧹 Cleaned dates & merged datasets.

🧩 Filled missing values (e.g., Fat from BMI).

⚡ Sampled heart rate data for performance.

🔠 Encoded categorical variables.

# 🧪 Feature Engineering

📅 Time: DayOfWeek, Month, IsWeekend.

🏋️ Activity: ActiveRatio, CaloriesPerStep, IntensityIndex.

😴 Sleep: SleepEfficiency, IsFragmentedSleep.

⚖️ BMI: Categories via custom function.

❤️ Heart Rate: mean & variability.

📊 Binned features for Calories, Steps, Sleep, HR.

# 🤖 Modeling & Evaluation
🔢 Regression

Models: LinearRegression, RandomForest, GradientBoosting, SVR, KNN, DecisionTree, Lasso, Ridge, SGD.
📏 Metrics: MAE, RMSE, R², MAPE.

🧩 Classification

Models: RandomForest, LogisticRegression, DecisionTree, SVM, Naive Bayes, KNN.
📏 Metrics: Accuracy, Precision, Recall, F1.
📊 Outputs: Confusion Matrices + Feature Importance plots.

# 📉 Visualizations

📦 Distributions (Steps, Calories, Sleep).

🔥 Heatmaps of correlations.

🟢 Scatter plots (Steps vs Calories, Sleep vs Activity).

📊 Actual vs Predicted graphs.

🌐 Interactive dashboards with Plotly.

# ✅ Deliverables

📑 Model performance tables.

💾 Saved trained models (.pkl).

🖼️ Visual insights & dashboards.

📓 Clean Jupyter Notebook.

📝 README documentation.

# 🌟 Strengths

End-to-end ML workflow 🔄.

Rich feature engineering ⚡.

Multi-model comparison 🧠.

Clear, engaging visualizations 🎨.

# 🔮 Future Improvements

🎛️ Hyperparameter tuning with GridSearchCV.

🧠 Interpretability via SHAP/LIME.

⚖️ Handle data imbalance.

🚀 Deployment via Flask/Streamlit dashboards.

🕒 Advanced time-series modeling (RNN/Transformers).

# 🛠️ Tools & Libraries

🐍 Python | 📦 pandas, numpy | 🤖 scikit-learn | 🎨 matplotlib, seaborn, plotly | 💾 joblib

# ✨ This project shows a complete ML workflow applied to real-world health data 🌍, combining data science, machine learning, and visualization to create actionable insights.


# 1. Imports

The code imports essential libraries:

joblib → for loading pre-trained ML models.

numpy & pandas → for handling data and input transformations.

# 2. Loading Regression Models

Two regression models are loaded:

calories_reg_model → predicts calories burned.

bmi_reg_model → predicts BMI.

# 3. Loading Classification Models

Several classification models are loaded, each trained for a specific prediction:

bmi_cls_model → predicts BMI category.

calories_cls_model → predicts whether calories are manually reported.

steps_cls_model → predicts whether the day is a weekend.

sleep_cls_model → predicts fragmented sleep.

sleep_eff_cls_model → predicts calories class (related to sleep efficiency).

activity_intensity_cls_model → predicts steps class (activity intensity).

hr_cls_model → predicts sleep class (based on heart rate).

# 4. Regression Functions

Functions that take input features and return numerical predictions:

predict_calories(features) → estimates calories burned using selected features like steps, distance, heart rate, etc.

predict_bmi(features) → predicts BMI based on features like weight, fat percentage, and calories per step.

# 5. Classification Functions

Functions that categorize input data into classes:

classify_bmi(features) → classifies BMI into categories.

classify_calories(features) → classifies whether calories are manually reported.

classify_steps(features) → classifies weekend vs. weekday steps.

classify_sleep(features) → classifies fragmented vs. normal sleep.

classify_sleep_efficiency(features) → classifies sleep efficiency.

classify_activity_intensity(features) → classifies activity intensity level.

classify_hr(features) → classifies sleep quality based on heart rate.

# 🏗️ 1. Imports

streamlit → for building the interactive web interface 🌐

ml_functions → to call the ML functions (e.g., predict_calories, classify_bmi…) 🤖

tkinter → for a simple desktop GUI 🖥️

joblib → to load pre-trained ML models 📂

# ⚡ 2. Streamlit Web Interface

st.set_page_config → sets up the page title and layout.

st.title → main title: “Health & Fitness AI Assistant 💡”.

st.selectbox → dropdown menu to choose a service (Regression or Classification).

# 🔮 3. Prediction Services (Regression)

Predict Calories 🔥
Takes input like:

Total steps 👣

Distance covered 📏

Heart rate ❤️

Active minutes ⏱️
→ Returns estimated calories burned.

Predict BMI ⚖️
Takes input: weight, height, fat %, calories per step.
→ Returns predicted BMI value.

# 📊 4. Classification Services

Classify BMI ⚖️ → predicts BMI category.

Classify Calories 🔥 → checks if calories are manually reported.

Classify Steps 👟 → classifies activity (e.g., weekend vs. weekday).

Classify Sleep 😴 → predicts sleep quality (normal vs. fragmented).

Classify Sleep Efficiency 🛌 → predicts sleep efficiency.

Classify Activity Intensity 💪 → classifies intensity level of activity.

Classify Heart Rate ❤️ → predicts heart rate category (e.g., normal/abnormal).

# 🖥️ 5. Tkinter Desktop App

A simple desktop interface with:

Input field ✍️

“Predict” button 🔘

Result shown in a popup message 📩

# 📌 In short: this project combines a Streamlit Web App 🌐 and a Tkinter Desktop App 🖥️ to create a smart Health AI Assistant 🤖💪, powered by ML models for both regression and classification.

# For run Gui You should write in terminal this comand " streamlit run Gui_St.py"
<img width="1908" height="925" alt="image" src="https://github.com/user-attachments/assets/13e81f9a-e5c2-45ed-b4c0-14c860b5bc22" />
<img width="1761" height="920" alt="image" src="https://github.com/user-attachments/assets/e46f4b25-db90-4adb-bd74-140f8614ba44" />

