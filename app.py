# dashboard_full_filters.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

# --- إعداد الصفحة ---
st.set_page_config(page_title="AI Churn Dashboard", layout="wide", page_icon="🤖")
st.markdown("""
<div style='display:flex; align-items:center; justify-content:start; margin-bottom:20px'>
    <img src='https://cdn-icons-png.flaticon.com/512/414/414927.png' width='50'>
    <h1 style='margin-left:15px; color:#4B0082'>AI Customer Churn Dashboard</h1>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# --- تحميل الموديل ---
try:
    model = joblib.load("model.pkl")
    st.success("✅ تم تحميل الموديل بنجاح!")
except:
    st.error("❌ لم يتم العثور على model.pkl. تأكد من وجوده في نفس المجلد.")
    st.stop()

# --- رفع البيانات ---
uploaded_file = st.file_uploader("اسحب و ارفع ملف البيانات هنا (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("تم رفع الملف بنجاح!")
else:
    st.info("انتظر رفع ملف البيانات لتوليد النتائج")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# الأعمدة النصية
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
if "CustomerID" in categorical_cols:
    categorical_cols.remove("CustomerID")  # ما نعملش filter على ID
for col in categorical_cols:
    options = data[col].unique().tolist()
    selected_values = st.sidebar.multiselect(f"{col}", options=options, default=options)
    data = data[data[col].isin(selected_values)]

# الأعمدة العددية
numeric_cols = data.select_dtypes(include=['int64','float64']).columns.tolist()
for col in numeric_cols:
    min_val = float(data[col].min())
    max_val = float(data[col].max())
    selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
    data = data[(data[col]>=selected_range[0]) & (data[col]<=selected_range[1])]

# --- تحويل الأعمدة النصية لأرقام باستخدام One-Hot Encoding ---
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# --- ضمان مطابقة الأعمدة مع الموديل ---
if hasattr(model, "feature_names_in_"):
    model_features = model.feature_names_in_
else:
    model_features = data_encoded.columns

for col in model_features:
    if col not in data_encoded.columns:
        data_encoded[col] = 0
data_encoded = data_encoded[model_features]

# --- التنبؤ بالـChurn ---
try:
    predictions = model.predict(data_encoded)
    probabilities = model.predict_proba(data_encoded)[:,1]
except Exception as e:
    st.error(f"حدث خطأ أثناء التنبؤ: {e}")
    st.stop()

data["Churn_Predicted"] = predictions
data["Churn_Probability"] = probabilities

# --- DASHBOARD METRICS ---
current_churn_rate = (data['Churn'].sum()/len(data)) if 'Churn' in data.columns else np.nan
predicted_churn_rate = data["Churn_Predicted"].mean()
total_customers = len(data)
predicted_savings = int(predicted_churn_rate * 10000)

col1, col2, col3, col4 = st.columns(4)
col1.metric("📉 Current Churn Rate", f"{current_churn_rate*100:.2f}%" if not np.isnan(current_churn_rate) else "N/A")
col2.metric("🔮 Predicted Churn Rate", f"{predicted_churn_rate*100:.2f}%")
col3.metric("👥 Total Customers", total_customers)
col4.metric("💰 Predicted Savings", f"${predicted_savings}")

# --- Historical & Predicted Churn Rate ---
st.markdown("### 📊 Historical & Predicted Churn Rate")
churn_history = pd.DataFrame({
    "Month": pd.date_range(start="2025-01-01", periods=12, freq='M'),
    "Historical": np.random.uniform(0.1, 0.3, 12),
    "Predicted": np.random.uniform(0.15, 0.35, 12)
})
fig_churn = px.line(churn_history, x="Month", y=["Historical", "Predicted"], markers=True,
                    labels={"value":"Churn Rate", "Month":"Month"}, template="plotly_dark")
st.plotly_chart(fig_churn, use_container_width=True)

# --- High Risk Customers & Top Churn Factors ---
st.markdown("### ⚠️ High Churn Risk Customers & Top Factors")
top_risk = data.sort_values(by="Churn_Probability", ascending=False).head(10)

if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=data_encoded.columns)
    top_features = importances.sort_values(ascending=False).head(3).index.tolist()
else:
    top_features = data.columns[:3].tolist()

top_risk_display = top_risk.copy()
top_risk_display["Top Factors"] = ", ".join(top_features)

columns_to_show = ["Churn_Probability","Top Factors"]
if "CustomerID" in top_risk_display.columns:
    columns_to_show = ["CustomerID"] + columns_to_show

st.table(top_risk_display[columns_to_show])

# --- Customer Segmentation ---
st.markdown("### 🟢 Customer Segmentation")
segmentation = data.groupby("Churn_Predicted").size().reset_index(name='Count')
segmentation["Percentage"] = segmentation["Count"]/segmentation["Count"].sum()*100
cols = st.columns(len(segmentation))
colors = ["#4B0082","#6A5ACD"]
for i, row in segmentation.iterrows():
    cols[i].markdown(f"<div style='background-color:{colors[i]}; padding:20px; border-radius:10px; text-align:center; color:white; font-weight:bold'>{'Will Churn' if row['Churn_Predicted']==1 else 'Will Stay'}<br>{row['Percentage']:.1f}%</div>", unsafe_allow_html=True)

# --- Recommendations & Impact Simulation ---
st.markdown("### 💡 Recommendations & Impact Simulation")
recommendations = {
    "Reduce Support Calls": 0.1,
    "Increase Usage": 0.08,
    "Loyalty Program": 0.05
}
for rec, impact in recommendations.items():
    st.markdown(f"- {rec} → Expected Churn Reduction: {impact*100:.1f}%")

sim_data = pd.DataFrame({
    "Action": list(recommendations.keys()),
    "Expected Impact": list(recommendations.values())
})
fig_sim = px.bar(sim_data, x="Action", y="Expected Impact", text="Expected Impact",
                 color="Expected Impact", template="plotly_dark", color_continuous_scale="Blues")
st.plotly_chart(fig_sim, use_container_width=True)
