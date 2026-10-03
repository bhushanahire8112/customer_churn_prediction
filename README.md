# Customer Churn Prediction

## 📌 Project Overview

This project predicts whether an e-commerce customer is likely to **stay** or **churn**.

The project uses Machine Learning classification algorithms and a Gradio interface for making predictions.

---

## 🎯 Objective

The main goal is to identify customers who may leave the service based on their:

- Usage behavior
- Shopping behavior
- Satisfaction
- Payment preferences
- Order history
- Customer information

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Gradio

---

## 📊 Dataset

The dataset contains customer information and their churn status.

### Target Column

`Churn`

- `0` → Stayed
- `1` → Churned

### Features Used

- Tenure
- PreferredLoginDevice
- CityTier
- WarehouseToHome
- PreferredPaymentMode
- Gender
- HourSpendOnApp
- NumberOfDeviceRegistered
- PreferedOrderCat
- SatisfactionScore
- MaritalStatus
- NumberOfAddress
- Complain
- OrderAmountHikeFromlastYear
- CouponUsed
- OrderCount
- DaySinceLastOrder
- CashbackAmount

---

## 🔄 Project Workflow

1. Load the dataset
2. Understand the data
3. Clean the data
4. Perform Exploratory Data Analysis (EDA)
5. Separate features and target
6. Split data into training and testing sets
7. Encode categorical features
8. Train Machine Learning models
9. Evaluate the models
10. Save the trained model
11. Create a prediction program
12. Build a Gradio UI

---

## 🤖 Models Tested

The following classification models were tested:

- Logistic Regression
- Random Forest
- Balanced Logistic Regression
- Gradient Boosting Classifier

### Model Results

| Model                        | Accuracy | Precision | Recall | F1 Score |
| ---------------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression          |   88.81% |    73.88% | 52.11% |   61.11% |
| Random Forest                |   88.81% |    73.88% | 52.11% |   61.11% |
| Balanced Logistic Regression |   79.22% |    43.96% | 84.21% |   57.76% |
| Gradient Boosting            |   91.47% |    80.92% | 64.74% |   71.93% |

Gradient Boosting was used as the final model based on the evaluation results from this project.

---

## 💾 Saved Files

Two important files are saved for making predictions:

- `churn_model.pkl` → Trained Gradient Boosting model
- `churn_preprocessor.pkl` → Preprocessing and encoding pipeline

---

## 🖥️ Gradio Interface

The project includes a Gradio UI where users can enter customer information and get a prediction.

The application gives one of two results:

```text
Customer is likely to Stay
```

or

```text
Customer is likely to Churn
```

---

## ▶️ How to Run

### 1. Install required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the prediction program

```bash
python predict.py
```

### 3. Run the Gradio application

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

---

## 📁 Project Structure

```text
customer_churn_prediction/
│
├── data/
│   └── customer_churn.csv
│
├── notebooks/
│   └── customer_churn_prediction.ipynb
│
├── churn_model.pkl
├── churn_preprocessor.pkl
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 📌 Example

### Input

Customer information is entered through the Gradio interface.

### Output

```text
Customer is likely to Stay ✅
```

---

## 🚀 Future Improvements

- Improve model performance
- Add probability of churn
- Add more visualizations
- Deploy the Gradio application
- Add more advanced models
