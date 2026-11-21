from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Retail Customer Sales Dashboard", page_icon="🛍️", layout="wide")
sns.set_theme(style="whitegrid")
BASE_DIR = Path(__file__).resolve().parent

st.title("🛍️ Retail Customer Sales Dashboard")
st.caption("Exploratory Data Analysis of Retail Customer Sales")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

st.sidebar.header("Dashboard Filters")
csv_files = list(BASE_DIR.glob("*.csv"))
if csv_files:
    data_file = next((f for f in csv_files if f.name.lower() == "retail_cleaned_dataset.csv"), csv_files[0])
    df = load_data(data_file)
    st.sidebar.success(f"Dataset: {data_file.name}")
else:
    uploaded = st.sidebar.file_uploader("Upload retail CSV dataset", type=["csv"])
    if uploaded is None:
        st.info("CSV ko GitHub repository mein app.py/Retail_EDA_Project.py ke saath upload karein, ya sidebar se upload karein.")
        st.stop()
    df = load_data(uploaded)

df = df.copy()

# Feature engineering
if {"Sales", "Profit"}.issubset(df.columns) and "Profit_Margin" not in df.columns:
    sales = pd.to_numeric(df["Sales"], errors="coerce")
    profit = pd.to_numeric(df["Profit"], errors="coerce")
    df["Profit_Margin"] = np.where(sales.ne(0), profit / sales * 100, np.nan)
if {"Order_Date", "Delivery_Date"}.issubset(df.columns) and "Delivery_Days" not in df.columns:
    od = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)
    dd = pd.to_datetime(df["Delivery_Date"], errors="coerce", dayfirst=True)
    df["Delivery_Days"] = (dd - od).dt.days
if {"Customer_ID", "Sales"}.issubset(df.columns) and "Revenue_per_Customer" not in df.columns:
    df["Revenue_per_Customer"] = df.groupby("Customer_ID")["Sales"].transform("sum")

# Filters
filtered = df.copy()
for col in ["Region", "Product_Category", "Customer_Segment", "Payment_Method", "Delivery_Mode"]:
    if col in filtered.columns:
        opts = sorted(filtered[col].dropna().astype(str).unique().tolist())
        chosen = st.sidebar.multiselect(col, opts, default=opts)
        filtered = filtered[filtered[col].astype(str).isin(chosen)]
if filtered.empty:
    st.warning("Selected filters ke liye data nahi mila.")
    st.stop()

# KPIs
st.header("📌 Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Records", f"{len(filtered):,}")
k1b = pd.to_numeric(filtered["Sales"], errors="coerce").sum() if "Sales" in filtered else None
k2.metric("Total Sales", f"{k1b:,.2f}" if k1b is not None else "N/A")
k2b = pd.to_numeric(filtered["Profit"], errors="coerce").sum() if "Profit" in filtered else None
k3.metric("Total Profit", f"{k2b:,.2f}" if k2b is not None else "N/A")
k4.metric("Duplicate Rows", f"{filtered.duplicated().sum():,}")

# 1 Dataset overview
st.header("1. Dataset Overview")
a, b = st.columns(2)
a.metric("Rows", filtered.shape[0])
b.metric("Columns", filtered.shape[1])
st.dataframe(filtered.head(), use_container_width=True)
with st.expander("View full dataset"):
    st.dataframe(filtered, use_container_width=True)
with st.expander("Data types"):
    st.dataframe(pd.DataFrame({"Column": filtered.columns, "Data Type": filtered.dtypes.astype(str)}), use_container_width=True)

# 2 Data quality
st.header("2. Data Quality Audit")
missing = pd.DataFrame({"Missing Values": filtered.isna().sum(), "Missing Percentage": (filtered.isna().mean()*100).round(2)})
a, b = st.columns(2)
a.metric("Missing Cells", int(filtered.isna().sum().sum()))
b.metric("Duplicate Rows", int(filtered.duplicated().sum()))
st.dataframe(missing, use_container_width=True)

# 3 Statistics
st.header("3. Descriptive Statistics")
numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
stats = pd.DataFrame()
if numeric_cols:
    stats = filtered[numeric_cols].describe().T
    stats["Median"] = filtered[numeric_cols].median()
    stats["Variance"] = filtered[numeric_cols].var()
    stats["IQR"] = filtered[numeric_cols].quantile(.75) - filtered[numeric_cols].quantile(.25)
    st.dataframe(stats.round(2), use_container_width=True)
    st.download_button("Download statistics", stats.round(2).to_csv().encode(), "descriptive_statistics.csv", "text/csv", key="download_statistics_section")
else:
    st.info("Numeric columns nahi mile.")

# 4 Univariate analysis
st.header("4. Univariate Analysis")
for col in ["Age", "Income", "Sales", "Profit"]:
    if col in filtered.columns:
        vals = pd.to_numeric(filtered[col], errors="coerce").dropna()
        if not vals.empty:
            st.subheader(f"{col} Distribution")
            st.write(vals.describe().round(2))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(vals, bins=30)
            ax.set(title=f"Distribution of {col}", xlabel=col, ylabel="Frequency")
            st.pyplot(fig)
            plt.close(fig)
for col in ["Region", "Product_Category", "Customer_Segment"]:
    if col in filtered.columns:
        st.subheader(f"{col} Frequency")
        counts = filtered[col].fillna("Missing").astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Transactions")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(counts.rename("Count").to_frame(), use_container_width=True)

# 5 Outliers
st.header("5. Outlier Detection (IQR Method)")
outlier_rows = []
for col in ["Sales", "Profit", "Age", "Income"]:
    if col in filtered.columns:
        vals = pd.to_numeric(filtered[col], errors="coerce").dropna()
        if vals.empty:
            continue
        q1, q3 = vals.quantile(.25), vals.quantile(.75)
        iqr = q3-q1
        low, high = q1-1.5*iqr, q3+1.5*iqr
        n = int(((vals < low) | (vals > high)).sum())
        outlier_rows.append({"Column": col, "Lower Limit": low, "Upper Limit": high, "Outlier Count": n})
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.boxplot(vals, vert=False)
        ax.set_title(f"Outlier Check - {col}")
        st.pyplot(fig)
        plt.close(fig)
if outlier_rows:
    outlier_df = pd.DataFrame(outlier_rows)
    st.dataframe(outlier_df.round(2), use_container_width=True)
else:
    outlier_df = pd.DataFrame()

# 6 Feature engineering
st.header("6. Feature Engineering")
features = [c for c in ["Transaction_ID", "Profit_Margin", "Delivery_Days", "Revenue_per_Customer"] if c in df.columns]
if features:
    st.dataframe(df[features].head(10), use_container_width=True)
else:
    st.info("Available columns se feature create nahi ho paya.")

# 7 Business analysis
st.header("7. Business Analysis")
business_tables = {}
if {"Region", "Sales"}.issubset(filtered.columns):
    business_tables["Total Sales by Region"] = filtered.groupby("Region")["Sales"].sum().sort_values(ascending=False)
if {"Product_Category", "Profit"}.issubset(filtered.columns):
    business_tables["Total Profit by Product Category"] = filtered.groupby("Product_Category")["Profit"].sum().sort_values(ascending=False)
if {"Customer_Segment", "Profit"}.issubset(filtered.columns):
    business_tables["Total Profit by Customer Segment"] = filtered.groupby("Customer_Segment")["Profit"].sum().sort_values(ascending=False)
if {"Payment_Method", "Sales"}.issubset(filtered.columns):
    business_tables["Sales by Payment Method"] = filtered.groupby("Payment_Method")["Sales"].sum().sort_values(ascending=False)
if "Delivery_Mode" in filtered.columns:
    aggs = {"Total_Orders": ("Transaction_ID" if "Transaction_ID" in filtered.columns else "Delivery_Mode", "count")}
    if "Delivery_Days" in filtered.columns:
        aggs["Average_Delivery_Days"] = ("Delivery_Days", "mean")
    business_tables["Delivery Mode Summary"] = filtered.groupby("Delivery_Mode").agg(**aggs).sort_values("Total_Orders", ascending=False)
if {"Region", "Sales"}.issubset(filtered.columns):
    business_tables["Average Sales by Region"] = filtered.groupby("Region")["Sales"].mean().sort_values(ascending=False)

if business_tables:
    for title, table in business_tables.items():
        st.subheader(title)
        if isinstance(table, pd.Series):
            table = table.to_frame()
        st.dataframe(table.round(2), use_container_width=True)
else:
    st.info("Business analysis ke liye required columns nahi mile.")

# 8 Comparison charts
st.header("8. Main Comparison Charts")
if {"Region", "Sales"}.issubset(filtered.columns):
    rs = filtered.groupby("Region")["Sales"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    rs.plot(kind="bar", ax=ax)
    ax.set_title("Total Sales by Region")
    st.pyplot(fig)
    plt.close(fig)
if {"Product_Category", "Profit"}.issubset(filtered.columns):
    cp = filtered.groupby("Product_Category")["Profit"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    cp.plot(kind="bar", ax=ax)
    ax.set_title("Total Profit by Product Category")
    st.pyplot(fig)
    plt.close(fig)

# 9 Correlation
st.header("9. Correlation Analysis")
if len(numeric_cols) >= 2:
    corr = filtered[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Correlation heatmap ke liye kam se kam 2 numeric columns chahiye.")

# 10 Downloads
st.header("10. Download Reports")
st.download_button("Download full dataset", df.to_csv(index=False).encode(), "retail_analysis_dataset.csv", "text/csv", key="download_full_dataset")
st.download_button("Download filtered dataset", filtered.to_csv(index=False).encode(), "filtered_retail_dataset.csv", "text/csv", key="download_filtered_dataset")
if not stats.empty:
    st.download_button("Download statistics", stats.round(2).to_csv().encode(), "descriptive_statistics.csv", "text/csv", key="download_statistics_report")
if not outlier_df.empty:
    st.download_button("Download outlier report", outlier_df.to_csv(index=False).encode(), "outlier_summary.csv", "text/csv", key="download_outlier_report")

st.success("✅ Retail EDA Dashboard Loaded Successfully!")
st.caption("Retail Customer Sales | Exploratory Data Analysis")
