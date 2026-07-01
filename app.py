
import io
import sqlite3
import pickle
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ChurnGuard AI", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model():
    with open("customer_churn_model.pkl","rb") as f:
        model = pickle.load(f)
    if hasattr(model, "threshold") and hasattr(model, "predict_proba"):
        return model
    return model
model = load_model()

@st.cache_data
def get_model_performance():
    return {
        "accuracy": 0.7508871540099361,
        "precision": 0.5210237659963437,
        "recall": 0.7620320855614974,
        "f1": 0.6188925081433225,
    }

conn = sqlite3.connect("users.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY,password TEXT)")
cur.execute("""CREATE TABLE IF NOT EXISTS history(
username TEXT,prediction TEXT,probability REAL,created_at TEXT
)""")
conn.commit()

if "logged_in" not in st.session_state:
    st.session_state.logged_in=False
if "username" not in st.session_state:
    st.session_state.username=""

def ensure_history_schema():
    cur.execute("PRAGMA table_info(history)")
    columns = [row[1] for row in cur.fetchall()]
    if "contract" not in columns:
        cur.execute("ALTER TABLE history ADD COLUMN contract TEXT")
    if "payment_method" not in columns:
        cur.execute("ALTER TABLE history ADD COLUMN payment_method TEXT")
    if "risk_level" not in columns:
        cur.execute("ALTER TABLE history ADD COLUMN risk_level TEXT")
    conn.commit()


def classify_risk(prob):
    prob_pct = prob * 100
    if prob_pct < 40:
        return "Low Risk", "#22c55e", "Customer is likely to stay."
    elif prob_pct <= 70:
        return "Medium Risk", "#f59e0b", "Customer needs attention."
    return "High Risk", "#ef4444", "Customer is likely to churn. Consider retention action."


def build_customer_insights(profile, risk_label):
    signals = []

    if profile.get("tenure", 0) <= 6:
        signals.append(
            {
                "headline": "Early-stage relationship",
                "reason": "The customer is still in the first months of service, so adoption and perceived value may need reinforcement.",
                "action": "Launch a guided onboarding sequence and highlight top-value features.",
            }
        )

    if profile.get("contract") == "Month-to-month":
        signals.append(
            {
                "headline": "Flexible contract behavior",
                "reason": "Short-term contract holders often evaluate alternatives more frequently, which can raise churn pressure.",
                "action": "Present a loyalty offer tied to a longer-term commitment.",
            }
        )

    if profile.get("monthly_charges", 0) >= 80:
        signals.append(
            {
                "headline": "Price sensitivity",
                "reason": "The customer is paying a relatively high monthly fee, so price-value balance may be a concern.",
                "action": "Offer a personalized plan review or targeted discount.",
            }
        )

    if profile.get("tech_support") in ["No", "No internet service"]:
        signals.append(
            {
                "headline": "Support-gap signal",
                "reason": "The profile suggests limited support coverage, which can weaken trust during service issues.",
                "action": "Introduce a support package with proactive assistance.",
            }
        )

    if profile.get("online_security") in ["No", "No internet service"]:
        signals.append(
            {
                "headline": "Security confidence gap",
                "reason": "The absence of security features may reduce confidence in the overall experience.",
                "action": "Highlight secure-service upgrades during the next outreach.",
            }
        )

    if profile.get("payment_method") == "Electronic check":
        signals.append(
            {
                "headline": "Payment-flow concern",
                "reason": "This payment pattern can reflect friction in the billing journey and increase churn exposure.",
                "action": "Provide billing support and simplify the payment experience.",
            }
        )

    if risk_label == "High Risk":
        signals = signals[:3] if len(signals) > 3 else signals
    elif risk_label == "Medium Risk":
        signals = signals[:2] if len(signals) > 2 else signals

    if not signals:
        reasons = ["The profile appears balanced, with no obvious friction points in the current customer data."]
        actions = ["Maintain service quality and keep engagement active to preserve confidence."]
        return reasons, actions

    reasons = [item["reason"] for item in signals]
    actions = [item["action"] for item in signals]
    return reasons, actions


def save_history(pred, prob, contract=None, payment_method=None, risk_level=None):
    cur.execute(
        "INSERT INTO history(username,prediction,probability,created_at,contract,payment_method,risk_level) VALUES(?,?,?,?,?,?,?)",
        (
            st.session_state.username,
            pred,
            float(prob),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            contract,
            payment_method,
            risk_level,
        ),
    )
    conn.commit()

ensure_history_schema()

if not st.session_state.logged_in:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #2563eb 100%); padding: 28px 24px; border-radius: 20px; margin-bottom: 22px; color: white;">
            <h1 style="margin:0 0 8px 0; font-size: 2rem;">🤖 ChurnGuard AI</h1>
            <p style="margin:0; font-size: 1.05rem; opacity: 0.95;">AI-powered customer retention system</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<p style='margin-top:-8px; color:#475569;'>Predict churn risk, understand customer behavior, and prioritize retention actions.</p>", unsafe_allow_html=True)
    mode=st.radio("Choose",["Login","Signup"],horizontal=True)

    auth_box = st.container()
    with auth_box:
        if mode=="Login":
            u=st.text_input("Username")
            p=st.text_input("Password",type="password")
            if st.button("Login", width="stretch"):
                cur.execute("SELECT * FROM users WHERE username=? AND password=?",(u,p))
                if cur.fetchone():
                    st.session_state.logged_in=True
                    st.session_state.username=u
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        else:
            u=st.text_input("Choose Username")
            p=st.text_input("Choose Password",type="password")
            c=st.text_input("Confirm Password",type="password")
            if st.button("Create Account", width="stretch"):
                if p!=c:
                    st.error("Passwords do not match")
                else:
                    cur.execute("SELECT * FROM users WHERE username=?",(u,))
                    if cur.fetchone():
                        st.warning("Username already exists")
                    else:
                        cur.execute("INSERT INTO users VALUES(?,?)",(u,p))
                        conn.commit()
                        st.success("Account created. Please login.")

else:
    st.sidebar.title("🤖 ChurnGuard AI")
    st.sidebar.success(f"Welcome {st.session_state.username}")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Prediction", "📊 Analytics", "📜 History", "ℹ️ About"],
    )
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    def get_history_df():
        return pd.read_sql(
            "SELECT * FROM history WHERE username=? ORDER BY created_at DESC",
            conn,
            params=(st.session_state.username,),
        )

    if page == "🏠 Dashboard":
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%); padding: 20px 22px; border-radius: 18px; border: 1px solid #dbeafe; margin-bottom: 16px;">
                <h2 style="margin:0 0 6px 0; color:#0f172a;">📊 Retention Command Center</h2>
                <p style="margin:0; color:#475569;">Monitor churn signals, customer risk trends, and actionable next steps in one place.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        history_df = get_history_df()

        if history_df.empty:
            st.info("No prediction history yet. Make a few predictions to see analytics and reports.")
        else:
            history_df["probability_pct"] = history_df["probability"] * 100
            history_df["risk_level"] = history_df["risk_level"].fillna(
                history_df["probability_pct"].apply(lambda p: classify_risk(p / 100)[0])
            )
            history_df["is_churn"] = history_df["prediction"].str.lower() == "churn"
            history_df["contract"] = history_df["contract"].fillna("Unknown")
            history_df["payment_method"] = history_df["payment_method"].fillna("Unknown")

            total_predictions = len(history_df)
            churn_count = int(history_df["is_churn"].sum())
            retained_count = total_predictions - churn_count
            churn_pct = (churn_count / total_predictions * 100) if total_predictions else 0
            high_risk_count = int((history_df["risk_level"] == "High Risk").sum())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total predictions", total_predictions)
            c2.metric("Total churn predicted", churn_count)
            c3.metric("Total customers retained", retained_count)
            c4.metric("Churn percentage", f"{churn_pct:.1f}%")
            c5.metric("High risk customers", high_risk_count)

            st.markdown("---")
            st.subheader("Quick Summary")
            st.success("Use the analytics and history pages to review trends, export reports, and monitor customer risk.")

    elif page == "🔮 Prediction":
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 20px 22px; border-radius: 18px; border: 1px solid #e2e8f0; margin-bottom: 16px;">
                <h2 style="margin:0 0 6px 0; color:#0f172a;">🔮 Customer Churn Prediction</h2>
                <p style="margin:0; color:#475569;">Enter customer profile details to evaluate churn risk and receive tailored retention guidance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

        with c2:
            tenure = st.number_input("Tenure", 0, 72, 12)
            monthly = st.number_input("Monthly Charges", 0.0, 10000.0, 70.0)
            total = st.number_input("Total Charges", 0.0, 100000.0, 1000.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                ],
            )

        with c3:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            if phone_service == "Yes":
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
            else:
                multiple_lines = "No phone service"

            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            if internet == "No":
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
            else:
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                online_backup = st.selectbox("Online Backup", ["No", "Yes"])
                device_protection = st.selectbox("Device Protection", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

        if st.button("Predict", width="stretch"):
            input_dict = {
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "tenure": tenure,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
                "gender_Male": 1 if gender == "Male" else 0,
                "Partner_Yes": 1 if partner == "Yes" else 0,
                "Dependents_Yes": 1 if dependents == "Yes" else 0,
                "PhoneService_Yes": 1 if phone_service == "Yes" else 0,
                "MultipleLines_No phone service": 1 if multiple_lines == "No phone service" else 0,
                "MultipleLines_Yes": 1 if multiple_lines == "Yes" else 0,
                "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
                "InternetService_No": 1 if internet == "No" else 0,
                "OnlineSecurity_No internet service": 1 if online_security == "No internet service" else 0,
                "OnlineSecurity_Yes": 1 if online_security == "Yes" else 0,
                "OnlineBackup_No internet service": 1 if online_backup == "No internet service" else 0,
                "OnlineBackup_Yes": 1 if online_backup == "Yes" else 0,
                "DeviceProtection_No internet service": 1 if device_protection == "No internet service" else 0,
                "DeviceProtection_Yes": 1 if device_protection == "Yes" else 0,
                "TechSupport_No internet service": 1 if tech_support == "No internet service" else 0,
                "TechSupport_Yes": 1 if tech_support == "Yes" else 0,
                "StreamingTV_No internet service": 1 if streaming_tv == "No internet service" else 0,
                "StreamingTV_Yes": 1 if streaming_tv == "Yes" else 0,
                "StreamingMovies_No internet service": 1 if streaming_movies == "No internet service" else 0,
                "StreamingMovies_Yes": 1 if streaming_movies == "Yes" else 0,
                "Contract_One year": 1 if contract == "One year" else 0,
                "Contract_Two year": 1 if contract == "Two year" else 0,
                "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
                "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
                "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
                "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
            }

            expected_features = list(model.feature_names_in_)
            for feature in expected_features:
                if feature not in input_dict:
                    input_dict[feature] = 0

            df = pd.DataFrame([input_dict])[expected_features]
            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1]
            text = "Churn" if pred == 1 else "Stay"
            prob_pct = prob * 100
            risk_label, risk_color, recommendation = classify_risk(prob)
            risk_icon = "🟢" if risk_label == "Low Risk" else "🟡" if risk_label == "Medium Risk" else "🔴"
            save_history(text, prob, contract=contract, payment_method=payment_method, risk_level=risk_label)

            profile = {
                "tenure": tenure,
                "monthly_charges": monthly,
                "contract": contract,
                "payment_method": payment_method,
                "tech_support": tech_support,
                "online_security": online_security,
            }
            reasons, actions = build_customer_insights(profile, risk_label)
            reason_html = "".join(f"<li style='margin-bottom:6px;'>{reason}</li>" for reason in reasons)
            action_html = "".join(f"<li style='margin-bottom:6px;'>{action}</li>" for action in actions)

            a, b, c = st.columns(3)
            a.metric("Prediction", text)
            b.metric("Probability", f"{prob_pct:.2f}%")
            c.metric("Risk Level", risk_label)

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 100%); padding:22px; border-radius:18px; border:1px solid #d9e2f0; margin-top:14px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
                        <div>
                            <h3 style="margin:0 0 6px 0; color:#0f172a;">{risk_icon} {risk_label}</h3>
                            <p style="margin:0; color:#475569;">{recommendation}</p>
                        </div>
                        <div style="font-size:34px; font-weight:700; color:{risk_color};">{prob_pct:.1f}%</div>
                    </div>
                    <div style="margin-top:16px; display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px;">
                        <div style="background:white; padding:14px; border-radius:12px; border:1px solid #e2e8f0;">
                            <h4 style="margin:0 0 8px 0; font-size:16px; color:#1e293b;">Customer Insight Drivers</h4>
                            <ul style="margin:0; padding-left:18px; color:#334155;">{reason_html}</ul>
                        </div>
                        <div style="background:white; padding:14px; border-radius:12px; border:1px solid #e2e8f0;">
                            <h4 style="margin:0 0 8px 0; font-size:16px; color:#1e293b;">Recommended Actions</h4>
                            <ul style="margin:0; padding-left:18px; color:#334155;">{action_html}</ul>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif page == "📊 Analytics":
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%); padding: 20px 22px; border-radius: 18px; border: 1px solid #e2e8f0; margin-bottom: 16px;">
                <h2 style="margin:0 0 6px 0; color:#0f172a;">📊 Analytics Dashboard</h2>
                <p style="margin:0; color:#475569;">A business-focused view of churn exposure, customer risk levels, and retention opportunities.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        history_df = get_history_df()

        if history_df.empty:
            st.info("No analytics data yet. Make predictions to generate insights.")
        else:
            history_df["probability_pct"] = history_df["probability"] * 100
            history_df["risk_level"] = history_df["risk_level"].fillna(
                history_df["probability_pct"].apply(lambda p: classify_risk(p / 100)[0])
            )
            history_df["is_churn"] = history_df["prediction"].str.lower() == "churn"
            history_df["contract"] = history_df["contract"].fillna("Unknown")
            history_df["payment_method"] = history_df["payment_method"].fillna("Unknown")

            total_customers = len(history_df)
            churn_count = int(history_df["is_churn"].sum())
            retained_count = total_customers - churn_count
            churn_pct = (churn_count / total_customers * 100) if total_customers else 0
            high_risk_count = int((history_df["risk_level"] == "High Risk").sum())

            summary_df = pd.DataFrame(
                {
                    "Metric": [
                        "Total customers analyzed",
                        "Churned customers",
                        "Retained customers",
                        "Churn rate",
                        "High risk customers",
                    ],
                    "Value": [
                        total_customers,
                        churn_count,
                        retained_count,
                        f"{churn_pct:.1f}%",
                        high_risk_count,
                    ],
                }
            )

            csv_buffer = io.BytesIO()
            summary_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.markdown("### Analytics Overview")
            st.caption("A polished view of churn exposure, customer risk segments, and retention opportunities.")

            metric_cols = st.columns(5, gap="small")
            cards = [
                ("Total Customers Analyzed", total_customers, "#0f172a"),
                ("Churned Customers", churn_count, "#dc2626"),
                ("Retained Customers", retained_count, "#16a34a"),
                ("Churn Rate %", f"{churn_pct:.1f}%", "#7c3aed"),
                ("High Risk Customers", high_risk_count, "#ea580c"),
            ]
            for col, (label, value, color) in zip(metric_cols, cards):
                with col:
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 16px 16px 14px 16px; border-radius: 14px; border: 1px solid #e2e8f0; min-height: 108px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);">
                            <div style="font-size: 0.84rem; color: #64748b; margin-bottom: 6px;">{label}</div>
                            <div style="font-size: 1.55rem; font-weight: 700; color: {color};">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("")
            export_col, insight_col = st.columns([1.1, 2.9], gap="small")
            with export_col:
                st.download_button(
                    "Download analytics summary report",
                    csv_buffer.getvalue(),
                    file_name=f"{st.session_state.username}_analytics_summary.csv",
                    mime="text/csv",
                )
            with insight_col:
                st.markdown(
                    "<div style='background:#f8fafc; padding:12px 14px; border-radius:12px; border:1px solid #e2e8f0; color:#475569;'>Business-focused analytics help prioritize retention campaigns and identify the most important churn signals.</div>",
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("### Churn Analysis")
            st.caption("Review customer outcome patterns using balanced, easy-to-scan charts.")

            chart_col_1, chart_col_2 = st.columns(2, gap="large")
            with chart_col_1:
                prediction_counts = (
                    history_df["prediction"]
                    .value_counts()
                    .reindex(["Churn", "Stay"])
                    .fillna(0)
                    .astype(int)
                )
                st.markdown("#### Prediction Mix")
                st.caption("A quick view of how many customers are currently flagged as churned versus retained.")
                st.bar_chart(
                    pd.DataFrame({"Customers": prediction_counts.values}, index=["Churned", "Retained"]),
                    height=260,
                )

            with chart_col_2:
                risk_counts = (
                    history_df["risk_level"]
                    .value_counts()
                    .reindex(["High Risk", "Medium Risk", "Low Risk"])
                    .fillna(0)
                    .astype(int)
                )
                st.markdown("#### Risk Distribution")
                st.caption("Highlights where the strongest retention attention is needed most urgently.")
                st.bar_chart(
                    pd.DataFrame({"Customers": risk_counts.values}, index=["High Risk", "Medium Risk", "Low Risk"]),
                    height=260,
                )

            chart_col_3, chart_col_4 = st.columns(2, gap="large")
            with chart_col_3:
                contract_summary = (
                    history_df.groupby("contract")
                    .agg(Total_Predictions=("prediction", "size"), Churn_Predictions=("is_churn", "sum"))
                    .reset_index()
                )
                contract_summary["Churn Percentage"] = (
                    (contract_summary["Churn_Predictions"] / contract_summary["Total_Predictions"] * 100).round(1)
                )
                st.markdown("#### Contract-Based Churn")
                st.caption("Shows which contract types are associated with the highest churn tendency.")
                st.bar_chart(contract_summary.set_index("contract")[["Churn Percentage"]], height=260)

            with chart_col_4:
                payment_summary = (
                    history_df.groupby("payment_method")
                    .agg(Total_Predictions=("prediction", "size"), Churn_Predictions=("is_churn", "sum"))
                    .reset_index()
                )
                payment_summary["Churn Percentage"] = (
                    (payment_summary["Churn_Predictions"] / payment_summary["Total_Predictions"] * 100).round(1)
                )
                st.markdown("#### Payment Method Churn")
                st.caption("Highlights payment methods that may be linked to customer churn behavior.")
                st.bar_chart(payment_summary.set_index("payment_method")[["Churn Percentage"]], height=260)

            st.divider()
            st.markdown("### Business Insights")
            st.caption("Key takeaways for retention planning and customer outreach.")
            observations = []
            if not contract_summary.empty:
                top_contract = contract_summary.sort_values("Churn Percentage", ascending=False).iloc[0]
                observations.append(f"{top_contract['contract']} contracts show the highest churn tendency at {top_contract['Churn Percentage']:.1f}%.")
            if not payment_summary.empty:
                top_payment = payment_summary.sort_values("Churn Percentage", ascending=False).iloc[0]
                observations.append(f"{top_payment['payment_method']} customers show the strongest churn signal at {top_payment['Churn Percentage']:.1f}%.")
            if high_risk_count:
                observations.append(f"{high_risk_count} customers currently sit in the high-risk group and may need immediate retention action.")
            if churn_pct > 0:
                observations.append(f"Overall churn rate stands at {churn_pct:.1f}%, making retention monitoring a priority.")
            if not observations:
                observations.append("No major churn pattern is visible yet; continue collecting predictions for stronger insight.")

            for obs in observations[:5]:
                st.markdown(f"- {obs}")

            with st.expander("Detailed analytics breakdown", expanded=False):
                st.markdown("#### Summary Table")
                st.dataframe(summary_df, width="stretch")

    elif page == "📜 History":
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 20px 22px; border-radius: 18px; border: 1px solid #e2e8f0; margin-bottom: 16px;">
                <h2 style="margin:0 0 6px 0; color:#0f172a;">📜 Prediction History</h2>
                <p style="margin:0; color:#475569;">Track previous predictions and keep a clear record of customer risk outcomes.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        hist = get_history_df()
        if hist.empty:
            st.info("No predictions yet.")
        else:
            csv_buffer = io.BytesIO()
            hist.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                "Download prediction history as CSV",
                csv_buffer.getvalue(),
                file_name=f"{st.session_state.username}_prediction_history.csv",
                mime="text/csv",
            )
            st.dataframe(hist, width="stretch")

    else:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%); padding: 24px 22px; border-radius: 20px; color:white; margin-bottom: 18px;">
                <h2 style="margin:0 0 8px 0;">ℹ️ About ChurnGuard AI</h2>
                <p style="margin:0; opacity:0.95;">A modern AI-powered customer retention platform built for fast, clear, and actionable churn decisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Problem Statement")
        st.write("Customer churn is costly and often difficult to detect early. Teams need a simple way to identify at-risk users before they leave.")
        st.subheader("Solution")
        st.write("ChurnGuard AI uses a trained machine learning model to estimate churn probability and highlight the most relevant retention signals for each customer profile.")
        st.subheader("ML Approach")
        st.write("The app uses a RandomForestClassifier trained on the Telco customer churn dataset to predict churn risk from structured customer attributes.")
        st.subheader("Model Performance")
        performance = get_model_performance()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{performance['accuracy'] * 100:.2f}%")
        with col2:
            st.metric("Precision", f"{performance['precision'] * 100:.2f}%")
        with col3:
            st.metric("Recall", f"{performance['recall'] * 100:.2f}%")
        with col4:
            st.metric("F1 Score", f"{performance['f1'] * 100:.2f}%")

        st.markdown("---")
        st.subheader("Model Information")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.markdown("**Algorithm**\nRandom Forest Classifier")
        with info_col2:
            st.markdown("**Problem Type**\nBinary Classification")
        with info_col3:
            st.markdown("**Target**\nCustomer Churn Prediction")

        st.subheader("Key Features")
        st.markdown(
            "- Real-time churn prediction with probability and risk scoring\n"
            "- Personalized customer insight recommendations\n"
            "- Historical prediction tracking in SQLite\n"
            "- Analytics dashboard with visual summaries\n"
            "- Exportable reports for presentations and reviews"
        )
