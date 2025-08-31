# 💓 Heart Stroke Prediction App 

This repository contains **two interactive Streamlit apps** for predicting heart stroke risk. Both apps use machine learning models and provide a user-friendly interface to assess heart health.

## App 1: Heart Stroke Prediction by Priyanka
**Description:**  
This app predicts the risk of heart stroke using a **K-Nearest Neighbors (KNN) model**. Users can input their health parameters and get an instant risk assessment.

**Features:**

- Predicts heart stroke risk (High / Low)
- Interactive sliders and dropdowns for input
- Handles both categorical and numerical features
- Scaled input for accurate predictions
- Clean UI with instant feedback

**Screenshot:**
![App 1 Screenshot](a1.png)
![App 2 Screenshot](a11.png)

**Usage:**

1. Open the app in your browser.  
2. Enter your health details: Age, Sex, Blood Pressure, Cholesterol, etc.  
3. Click **Predict** to see your risk.  
4. High risk shows a warning; low risk shows a success message.

---

## App 2: Heart Stroke Prediction with Visual Insights

**Description:**  
This app expands on App 1 by providing **visual insights** into the prediction, such as charts and probability scores. It uses the same KNN model but adds a **more interactive experience**.

**Features:**

- Risk prediction (High / Low)
- Probability score display
- Interactive charts (Plotly) for better visualization
- User-friendly input forms
- Responsive design

**Screenshot:**


**Usage:**

1. Run the app using Streamlit.  
2. Input health parameters.  
3. Click **Predict** to see risk and probability score.  
4. Explore charts for additional insights.


## Demo

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/heart-stroke-prediction.git
cd heart-stroke-prediction


2. Install dependencies:
```bash
pip install -r requirements.txt


3. Run the app:
```bash
python -m streamlit run app.py

python -m streamlit run app2.py

