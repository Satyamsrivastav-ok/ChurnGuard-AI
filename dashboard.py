
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- LOAD MODEL ----------------
def show_dashboard():

    model = pickle.load(
        open("notebooks/customer_churn_model.pkl","rb")
    )


# ---------------- PAGE SETTINGS ----------------

    st.set_page_config(
        page_title="Customer Churn Predictor",
        page_icon="📊",
        layout="wide"
    )


    # ---------------- TITLE ----------------

    st.title("📊 ChurnGuard AI")
    col1, col2 = st.columns([3,1])

    with col1:
        st.info("👋 Welcome to ChurnGuard AI Dashboard")

    with col2:
        st.success("🟢 Model Status : Active")

    st.markdown("""
    ### AI-Powered Customer Retention Intelligence Platform
    Predict customer churn risk and take proactive retention actions.
    """)

    st.divider()
    st.markdown(
    """
    This AI model predicts whether a customer is likely to leave the service.
    """
    )


    # ---------------- SIDEBAR ----------------

    st.sidebar.title("📋 Customer Details")
    st.sidebar.caption("Enter customer information below")
    st.sidebar.divider()

    gender = st.sidebar.selectbox(
        "Gender",
        ["Male","Female"]
    )


    senior = st.sidebar.selectbox(
        "Senior Citizen",
        [0,1]
    )


    partner = st.sidebar.selectbox(
        "Partner",
        ["Yes","No"]
    )


    dependents = st.sidebar.selectbox(
        "Dependents",
        ["Yes","No"]
    )


    tenure = st.sidebar.slider(
        "Tenure (Months)",
        0,72,12
    )


    monthly = st.sidebar.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        50.0
    )


    total = st.sidebar.number_input(
        "Total Charges",
        0.0,
        10000.0,
        500.0
    )


    contract = st.sidebar.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


    internet = st.sidebar.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )



    # ---------------- PREDICTION ----------------


    if st.sidebar.button(
        "🔍 Predict Customer Risk",
        use_container_width=True
    ):
        data = pd.DataFrame({

            "gender":[gender],
            "SeniorCitizen":[senior],
            "Partner":[partner],
            "Dependents":[dependents],
            "tenure":[tenure],
            "MonthlyCharges":[monthly],
            "TotalCharges":[total],
            "Contract":[contract],
            "InternetService":[internet]

        })


        # same preprocessing

        data = pd.get_dummies(
            data,
            drop_first=True
        )


        # align columns

        cols = model.feature_names_in_


        for c in cols:
            if c not in data.columns:
                data[c]=0


        data=data[cols]


        prediction=model.predict(data)[0]


        probability=model.predict_proba(data)[0][1]



        # ---------------- RESULT ----------------


        st.divider()


        col1,col2,col3 = st.columns(3)



        with col1:

            st.metric(
                "Prediction",
                "Churn" if prediction==1 else "Stay"
            )


        with col2:

            st.metric(
                "Churn Probability",
                f"{probability*100:.2f}%"
            )


        with col3:

            risk = ""

            if probability <0.3:
                risk="LOW"

            elif probability <0.6:
                risk="MEDIUM"

            else:
                risk="HIGH"


            st.metric(
                "Risk Level",
                risk
            )


        st.divider()



        if prediction==1:

            st.error(
            """
            ⚠️ High Risk Customer

            Recommendation:
            - Offer retention benefits
            - Provide discounts
            - Improve customer support
            """
            )


        else:

            st.success(
            """
            ✅ Customer is Stable

            Recommendation:
            - Maintain service quality
            - Offer loyalty rewards
            """
            )
            st.divider()

    st.subheader("📌 Feature Importance")


    importance = pd.DataFrame({

        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_

    })


    importance = importance.sort_values(
        "Importance",
        ascending=False
    )


    st.bar_chart(
        importance.set_index("Feature").head(10)
    )
