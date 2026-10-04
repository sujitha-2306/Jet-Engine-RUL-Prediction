# ✈️ Jet Engine RUL Prediction

A machine learning project for predicting the **Remaining Useful Life (RUL)** of aircraft jet engines using sensor data from the **NASA C-MAPSS FD001 dataset**.

The project uses a **Gradient Boosting Regression model** to estimate engine RUL and provides an interactive **Streamlit dashboard** for monitoring engine health, predicted RUL, risk status, and fleet-level insights.

## 📌 Project Overview

Predictive maintenance helps identify potential engine degradation before failure occurs. Instead of waiting for an engine to fail, this project estimates how many operational cycles remain.

The system:

* Processes aircraft engine sensor data
* Calculates RUL for training and test data
* Analyzes sensor relationships with RUL
* Trains and evaluates machine learning models
* Predicts RUL for test engines
* Provides engine health and risk insights
* Displays predictions through an interactive Streamlit dashboard

## 🎯 Objectives

* Predict the Remaining Useful Life of aircraft engines
* Explore relationships between sensor measurements and engine degradation
* Compare machine learning regression approaches
* Build a practical predictive-maintenance application
* Provide an interactive dashboard for engine monitoring

## 🗂️ Dataset

This project uses the **NASA C-MAPSS FD001** turbofan engine degradation simulation dataset.

The dataset contains:

* Engine operating cycles
* 3 operating settings
* 21 sensor measurements
* Multiple engine units with different degradation patterns

### Dataset Files

```text
data/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

Each engine is observed over multiple operating cycles. The RUL value represents the number of cycles remaining before the end of the observed degradation period.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**
* **Joblib**

## 🤖 Machine Learning

The following regression models were explored:

### Random Forest Regressor

Validation Results:

* MAE: **23.95 cycles**
* RMSE: **31.61 cycles**
* R²: **0.768**

### Gradient Boosting Regressor

The final model uses:

```text
n_estimators = 200
learning_rate = 0.05
max_depth = 3
random_state = 42
```

RUL values were capped at **125 cycles** to make the model more focused on the useful maintenance range.

### Final Engine-Level Results

Evaluation was performed on the latest observed cycle of each of the **100 test engines**.

| Metric |           Result |
| ------ | ---------------: |
| MAE    | **11.65 cycles** |
| RMSE   | **16.07 cycles** |
| R²     |        **0.839** |

These results represent the evaluation setup used in this project and are **not official NASA benchmark scores**.

## 📊 Important Sensors

Correlation analysis identified several sensors with stronger relationships with RUL in the training data.

Top correlated sensors:

1. `sensor_11`
2. `sensor_4`
3. `sensor_12`
4. `sensor_7`
5. `sensor_15`

Correlation alone does not imply that these sensors are individually responsible for engine degradation; they are simply among the stronger relationships observed in this dataset.

## 🖥️ Streamlit Dashboard

The project includes an interactive dashboard called **Jet Engine Intelligence**.

The dashboard provides:

* ✈️ Engine selection
* 🔄 Current operating cycle
* 📉 Predicted RUL
* ❤️ Engine health percentage
* ⚠️ Risk status
* 🔧 Maintenance recommendation
* 📊 Sensor diagnostics
* 📈 Fleet RUL distribution
* 🤖 Model performance metrics

## 📁 Project Structure

```text
Jet-Engine-RUL-Prediction/
│
├── app.py
├── main.py
├── requirements.txt
├── rul_prediction_model.pkl
├── final_rul_predictions.csv
│
└── data/
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Jet-Engine-RUL-Prediction.git
```

Move into the project directory:

```bash
cd Jet-Engine-RUL-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 🚀 Run the Streamlit Dashboard

Run:

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser.

## 🔍 How It Works

```text
NASA C-MAPSS Dataset
        ↓
Data Preprocessing
        ↓
RUL Calculation
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Gradient Boosting Model
        ↓
RUL Prediction
        ↓
Engine Health & Risk Analysis
        ↓
Streamlit Dashboard
```

## 📌 Future Improvements

* Add time-series based deep learning models such as LSTM
* Add more C-MAPSS datasets such as FD002, FD003, and FD004
* Improve degradation trend visualization
* Add historical engine-cycle analysis
* Add automated maintenance alerts
* Deploy the Streamlit application online

## 👩‍💻 Author

**Sujitha Rajavel**

B.Tech Artificial Intelligence & Data Science

GitHub: https://github.com/sujitha-2306

---

⭐ If you find this project useful, consider giving the repository a star!
