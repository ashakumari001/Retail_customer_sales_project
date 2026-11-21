
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Retail Customer Sales Dashboard",
    page_icon="🛍️",
    layout="wide"
)

sns.set_theme(style="whitegrid")

BASE_DIR = Path(__file__).resolve().parent

st.title("🛍️ Retail Customer Sales Dashboard")
st.markdown("### Exploratory Data Analysis of Retail Customer Sales")
st.markdown("---")


# ---------------- LOAD DATASET ----------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


st.sidebar.title("Dashboard Filters")

# Automatically detect CSV file
csv_files = list(BASE_DIR.glob("*.csv"))

if csv_files:
    DATA_FILE = next(
        (f for f in csv_files if f.name == "Retail_Cleaned_Dataset.csv"),
        csv_files[0]
    )
    df = load_data(DATA_FILE)
    st.sidebar.success(f"Dataset: {DATA_FILE.name}")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Retail CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is None:
        st.warning(
            "CSV dataset nahi mili. GitHub repository mein CSV file upload karein "
            "ya sidebar se dataset upload karein."
        )
        st.stop()

    df = load_data(uploaded_file)

df = df.copy()


# ---------------- FEATURE ENGINEERING ----------------
if "Profit_Margin" not in df.columns and {"Profit", "Sales"}.issubset(df.columns):
    df["Profit_Margin"] = np.where(
        df["Sales"] != 0,
        (df["Profit"] / df["Sales"]) * 100,
        np.nan
    )

if "Delivery_Days" not in df.columns and {
    "Order_Date", "Delivery_Date"
}.issubset(df.columns):

    order = pd.to_datetime(
        df["Order_Date"], errors="coerce", dayfirst=True
    )
    delivery = pd.to_datetime(
        df["Delivery_Date"], errors="coerce", dayfirst=True
    )

    df["Delivery_Days"] = (delivery - order).dt.days

if "Revenue_per_Customer" not in df.columns and {
    "Customer_ID", "Sales"
}.issubset(df.columns):

    df["Revenue_per_Customer"] = (
        df.groupby("Customer_ID")["Sales"].transform("sum")
    )


# ---------------- SIDEBAR FILTERS ----------------
filtered_df = df.copy()

filter_columns = [
    "Region",
    "Product_Category",
    "Customer_Segment",
    "Payment_Method",
    "Delivery_Mode"
]

for col in filter_columns:
    if col in filtered_df.columns:

        options = sorted(
            filtered_df[col].dropna().astype(str).unique()
        )

        selected = st.sidebar.multiselect(
            f"Select {col}",
            options,
            default=options
        )

        filtered_df = filtered_df[
            filtered_df[col].astype(str).isin(selected)
        ]

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()


# ---------------- KPI CARDS ----------------
st.header("📌 Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(filtered_df):,}")

if "Sales" in filtered_df.columns:
    c2.metric("Total Sales", f"{filtered_df['Sales'].sum():,.2f}")
else:
    c2.metric("Total Sales", "N/A")

if "Profit" in filtered_df.columns:
    c3.metric("Total Profit", f"{filtered_df['Profit'].sum():,.2f}")
else:
    c3.metric("Total Profit", "N/A")

c4.metric(
    "Duplicate Rows",
    f"{filtered_df.duplicated().sum():,}"
)

st.markdown("---")


# ---------------- 1. DATASET OVERVIEW ----------------
st.header("1. 📊 Dataset Overview")

col1, col2 = st.columns(2)

col1.metric("Rows", filtered_df.shape[0])
col2.metric("Columns", filtered_df.shape[1])

st.subheader("First Five Records")
st.dataframe(filtered_df.head(), use_container_width=True)

with st.expander("View Complete Dataset"):
    st.dataframe(filtered_df, use_container_width=True)

with st.expander("Data Types"):
    dtype_df = pd.DataFrame({
        "Column": filtered_df.columns,
        "Data Type": filtered_df.dtypes.astype(str).values
    })
    st.dataframe(dtype_df, use_container_width=True)


# ---------------- 2. DATA QUALITY AUDIT ----------------
st.header("2. 🔍 Data Quality Audit")

missing = pd.DataFrame({
    "Missing Values": filtered_df.isnull().sum(),
    "Missing Percentage": (
        filtered_df.isnull().mean() * 100
    ).round(2)
})

col1, col2 = st.columns(2)

col1.metric(
    "Total Missing Values",
    int(filtered_df.isnull().sum().sum())
)

col2.metric(
    "Duplicate Records",
    int(filtered_df.duplicated().sum())
)

st.subheader("Missing Value Report")
st.dataframe(missing, use_container_width=True)


# ---------------- 3. DESCRIPTIVE STATISTICS ----------------
st.header("3. 📈 Descriptive Statistics")

numeric_cols = filtered_df.select_dtypes(
    include=np.number
).columns.tolist()

if numeric_cols:

    stats = filtered_df[numeric_cols].describe().T

    stats["Median"] = filtered_df[numeric_cols].median()
    stats["Variance"] = filtered_df[numeric_cols].var()

    stats["IQR"] = (
        filtered_df[numeric_cols].quantile(0.75)
        - filtered_df[numeric_cols].quantile(0.25)
    )

    st.dataframe(stats.round(2), use_container_width=True)

    st.download_button(
        "Download Descriptive Statistics",
        stats.round(2).to_csv().encode("utf-8"),
        "descriptive_statistics.csv",
        "text/csv"
    )

else:
    st.info("No numeric columns available.")


# ---------------- 4. UNIVARIATE ANALYSIS ----------------
st.header("4. 📊 Univariate Analysis")

for col in ["Age", "Income", "Sales", "Profit"]:

    if col in filtered_df.columns:

        st.subheader(f"{col} Distribution")

        st.write(
            filtered_df[col].describe().round(2)
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(
            filtered_df[col].dropna(),
            bins=30
        )

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)


for col in ["Region", "Product_Category", "Customer_Segment"]:

    if col in filtered_df.columns:

        st.subheader(f"{col} Frequency")

        counts = (
            filtered_df[col]
            .fillna("Missing")
            .astype(str)
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        counts.plot(kind="bar", ax=ax)

        ax.set_title(f"Count of {col}")
        ax.set_ylabel("Number of Transactions")

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            counts.rename("Count").to_frame(),
            use_container_width=True
        )


# ---------------- 5. OUTLIER DETECTION ----------------
st.header("5. 🚨 Outlier Detection (IQR Method)")

outlier_rows = []

for col in ["Sales", "Profit", "Age", "Income"]:

    if col in filtered_df.columns:

        values = pd.to_numeric(
            filtered_df[col], errors="coerce"
        ).dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((values < lower) | (values > upper)).sum()
        )

        outlier_rows.append({
            "Column": col,
            "Lower Limit": lower,
            "Upper Limit": upper,
            "Outlier Count": count
        })

        fig, ax = plt.subplots(figsize=(8, 3))

        ax.boxplot(values, vert=False)
        ax.set_title(f"Outlier Check - {col}")

        st.pyplot(fig)
        plt.close(fig)

if outlier_rows:

    outlier_df = pd.DataFrame(outlier_rows)

    st.subheader("Outlier Summary")
    st.dataframe(
        outlier_df.round(2),
        use_container_width=True
    )


# ---------------- 6. FEATURE ENGINEERING ----------------
st.header("6. ⚙️ Feature Engineering")

feature_cols = [
    col for col in [
        "Transaction_ID",
        "Profit_Margin",
        "Delivery_Days",
        "Revenue_per_Customer"
    ]
    if col in df.columns
]

if feature_cols:

    st.subheader("Newly Created Features")

    st.dataframe(
        df[feature_cols].head(10),
        use_container_width=True
    )

else:
    st.info("Feature columns could not be created from available data.")


# ---------------- 7. BUSINESS QUESTIONS ----------------
st.header("7. 💼 Business Analysis")

business_tables = {}

if {"Region", "Sales"}.issubset(filtered_df.columns):

    business_tables["Total Sales by Region"] = (
        filtered_df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

if {"Product_Category", "Profit"}.issubset(filtered_df.columns):

    business_tables["Total Profit by Product Category"] = (
        filtered_df.groupby("Product_Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

if {"Customer_Segment", "Profit"}.issubset(filtered_df.columns):

    business_tables["Total Profit by Customer Segment"] = (
        filtered_df.groupby("Customer_Segment")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

if {"Payment_Method", "Sales"}.issubset(filtered_df.columns):

    business_tables["Sales by Payment Method"] = (
        filtered_df.groupby("Payment_Method")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

if "Delivery_Mode" in filtered_df.columns:

    order_col = (
        "Transaction_ID"
        if "Transaction_ID" in filtered_df.columns
        else "Delivery_Mode"
    )

    delivery_summary = filtered_df.groupby("Delivery_Mode").agg(
        Total_Orders=(order_col, "count")
    )

    if "Delivery_Days" in filtered_df.columns:

        delivery_summary["Average_Delivery_Days"] = (
            filtered_df.groupby("Delivery_Mode")["Delivery_Days"].mean()
        )

    business_tables["Delivery Mode Summary"] = (
        delivery_summary.sort_values("Total_Orders", ascending=False)
    )

if {"Region", "Sales"}.issubset(filtered_df.columns):

    business_tables["Average Sales by Region"] = (
        filtered_df.groupby("Region")["Sales"]
        .mean()
        .sort_values(ascending=False)
    )

if business_tables:

    for title, table in business_tables.items():

        st.subheader(title)

        st.dataframe(
            table.round(2).to_frame(),
            use_container_width=True
        )

else:
    st.info("Required business analysis columns not found.")


# ---------------- 8. MAIN COMPARISON CHARTS ----------------
st.header("8. 📉 Main Comparison Charts")

if {"Region", "Sales"}.issubset(filtered_df.columns):

    st.subheader("Total Sales by Region")

    region_sales = (
        filtered_df.groupby("Region")["Sales"].sum().sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    region_sales.plot(kind="bar", ax=ax)

    ax.set_ylabel("Sales")
    ax.set_xlabel("Region")

    st.pyplot(fig)
    plt.close(fig)


if {"Product_Category", "Profit"}.issubset(filtered_df.columns):

    st.subheader("Total Profit by Product Category")

    category_profit = (
        filtered_df.groupby("Product_Category")["Profit"].sum().sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    category_profit.plot(kind="bar", ax=ax)

    ax.set_ylabel("Profit")
    ax.set_xlabel("Product Category")

    st.pyplot(fig)
    plt.close(fig)


# ---------------- 9. CORRELATION ANALYSIS ----------------
st.header("9. 🔗 Correlation Analysis")

if len(numeric_cols) >= 2:

    corr = filtered_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("At least two numeric columns are required.")


# ---------------- 10. DOWNLOAD REPORTS ----------------
st.header("10. 📥 Download Reports")

st.download_button(
    "Download Analysis Dataset",
    df.to_csv(index=False).encode("utf-8"),
    "retail_analysis_dataset.csv",
    "text/csv"
)

st.download_button(
    "Download Filtered Dataset",
    filtered_df.to_csv(index=False).encode("utf-8"),
    "filtered_retail_dataset.csv",
    "text/csv"
)

if numeric_cols:

    st.download_button(
        "Download Statistics",
        stats.round(2).to_csv().encode("utf-8"),
        "descriptive_statistics.csv",
        "text/csv"
    )

if outlier_rows:

    st.download_button(
        "Download Outlier Report",
        pd.DataFrame(outlier_rows).to_csv(index=False).encode("utf-8"),
        "outlier_summary.csv",
        "text/csv"
    )

st.markdown("---")

st.success("✅ Retail EDA Dashboard Loaded Successfully!")

st.caption("Retail Customer Sales | Exploratory Data Analysis")
