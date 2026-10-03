import gradio as gr
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("ecommerce/model/churn_model.pkl")
preprocessor = joblib.load("ecommerce/model/churn_preprocessor.pkl")


def predict_churn(
    tenure,
    preferred_login_device,
    city_tier,
    warehouse_to_home,
    preferred_payment_mode,
    gender,
    hour_spend_on_app,
    number_of_device_registered,
    prefered_order_cat,
    satisfaction_score,
    marital_status,
    number_of_address,
    complain,
    order_amount_hike,
    coupon_used,
    order_count,
    day_since_last_order,
    cashback_amount
):

    new_customer = pd.DataFrame([{
        "Tenure": tenure,
        "PreferredLoginDevice": preferred_login_device,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_to_home,
        "PreferredPaymentMode": preferred_payment_mode,
        "Gender": gender,
        "HourSpendOnApp": hour_spend_on_app,
        "NumberOfDeviceRegistered": number_of_device_registered,
        "PreferedOrderCat": prefered_order_cat,
        "SatisfactionScore": satisfaction_score,
        "MaritalStatus": marital_status,
        "NumberOfAddress": number_of_address,
        "Complain": complain,
        "OrderAmountHikeFromlastYear": order_amount_hike,
        "CouponUsed": coupon_used,
        "OrderCount": order_count,
        "DaySinceLastOrder": day_since_last_order,
        "CashbackAmount": cashback_amount
    }])

    # Preprocess
    new_encoded = preprocessor.transform(new_customer)

    if hasattr(new_encoded, "toarray"):
        new_encoded = new_encoded.toarray()

    # Prediction
    prediction = model.predict(new_encoded)

    if prediction[0] == 1:
        return "⚠️ Customer is likely to Churn"
    else:
        return "✅ Customer is likely to Stay"


# ---------------- UI ----------------

with gr.Blocks(title="Customer Churn Prediction") as demo:

    gr.Markdown(
        """
        # 🛍️ Customer Churn Prediction
        ### Predict whether a customer is likely to stay or churn
        Enter the customer information below and click **Predict Churn**.
        """
    )

    with gr.Row():

        with gr.Column():
            gr.Markdown("### 👤 Customer Information")

            tenure = gr.Number(
                label="Tenure",
                value=4
            )

            preferred_login_device = gr.Dropdown(
                ["Mobile Phone", "Computer", "Phone"],
                label="Preferred Login Device",
                value="Mobile Phone"
            )

            city_tier = gr.Number(
                label="City Tier",
                value=1
            )

            warehouse_to_home = gr.Number(
                label="Warehouse to Home",
                value=6
            )

            preferred_payment_mode = gr.Dropdown(
                ["Debit Card", "Credit Card", "E wallet", "UPI",
                 "Cash on Delivery"],
                label="Preferred Payment Mode",
                value="Debit Card"
            )

            gender = gr.Dropdown(
                ["Male", "Female"],
                label="Gender",
                value="Male"
            )

            marital_status = gr.Dropdown(
                ["Single", "Married", "Divorced"],
                label="Marital Status",
                value="Single"
            )

            satisfaction_score = gr.Number(
                label="Satisfaction Score",
                value=3
            )

            complain = gr.Number(
                label="Complain (0 = No, 1 = Yes)",
                value=0
            )

        with gr.Column():
            gr.Markdown("### 🛒 Shopping Information")

            hour_spend_on_app = gr.Number(
                label="Hour Spend on App",
                value=3
            )

            number_of_device_registered = gr.Number(
                label="Number of Devices Registered",
                value=3
            )

            prefered_order_cat = gr.Dropdown(
                [
                    "Laptop & Accessory",
                    "Mobile",
                    "Mobile Phone",
                    "Fashion",
                    "Grocery",
                    "Others"
                ],
                label="Preferred Order Category",
                value="Laptop & Accessory"
            )

            number_of_address = gr.Number(
                label="Number of Address",
                value=2
            )

            order_amount_hike = gr.Number(
                label="Order Amount Hike From Last Year",
                value=15
            )

            coupon_used = gr.Number(
                label="Coupon Used",
                value=1
            )

            order_count = gr.Number(
                label="Order Count",
                value=2
            )

            day_since_last_order = gr.Number(
                label="Days Since Last Order",
                value=3
            )

            cashback_amount = gr.Number(
                label="Cashback Amount",
                value=150
            )

    gr.Markdown("### 🔮 Prediction")

    with gr.Row():
        predict_button = gr.Button(
            "🔍 Predict Churn",
            variant="primary"
        )

        clear_button = gr.ClearButton(
            value="🗑️ Clear"
        )

    output = gr.Textbox(
        label="Prediction Result",
        interactive=False
    )

    predict_button.click(
        fn=predict_churn,
        inputs=[
            tenure,
            preferred_login_device,
            city_tier,
            warehouse_to_home,
            preferred_payment_mode,
            gender,
            hour_spend_on_app,
            number_of_device_registered,
            prefered_order_cat,
            satisfaction_score,
            marital_status,
            number_of_address,
            complain,
            order_amount_hike,
            coupon_used,
            order_count,
            day_since_last_order,
            cashback_amount
        ],
        outputs=output
    )


demo.launch()