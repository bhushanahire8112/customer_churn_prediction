import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("ecommerce/model/churn_model.pkl")
preprocessor = joblib.load("ecommerce/model/churn_preprocessor.pkl")

# New customer data
new_customer = pd.DataFrame([{
    'Tenure': 4,
    'PreferredLoginDevice': 'Mobile Phone',
    'CityTier': 1,
    'WarehouseToHome': 6,
    'PreferredPaymentMode': 'Debit Card',
    'Gender': 'Male',
    'HourSpendOnApp': 3,
    'NumberOfDeviceRegistered': 3,
    'PreferedOrderCat': 'Laptop & Accessory',
    'SatisfactionScore': 3,
    'MaritalStatus': 'Single',
    'NumberOfAddress': 2,
    'Complain': 0,
    'OrderAmountHikeFromlastYear': 15,
    'CouponUsed': 1,
    'OrderCount': 2,
    'DaySinceLastOrder': 3,
    'CashbackAmount': 150
}])

# Preprocess the new customer
new_encoded = preprocessor.transform(new_customer)

# Convert sparse matrix to normal array if needed
if hasattr(new_encoded, "toarray"):
    new_encoded = new_encoded.toarray()

# Make prediction
prediction = model.predict(new_encoded)

# Display result
if prediction[0] == 1:
    print("Customer is likely to Churn ❌")
else:
    print("Customer is likely to Stay ✅")