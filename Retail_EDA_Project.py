#!/usr/bin/env python
import streamlit as st
# coding: utf-8

# In[ ]:


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Retail EDA Dashboard", layout="wide")

st.title("🛍️ Retail Customer Sales Dashboard")
st.write("Exploratory Data Analysis of Retail Customer Sales")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Retail_Cleaned_Dataset.csv"
OUTPUT_DIR = BASE_DIR / "EDA_Output"
OUTPUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")


def main():
    print("=" * 65)
    print("RETAIL CUSTOMER SALES - EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    if not DATA_FILE.exists():
        print("Dataset file not found:", DATA_FILE.name)
        print("Keep Retail_Cleaned_Dataset.csv in the same folder as this script.")
        return

    # 1. Load dataset
    df = pd.read_csv(DATA_FILE)
    print("\n1. DATASET OVERVIEW")
    print("Rows and columns:", df.shape)
    print("\nFirst five records:\n", df.head().to_string(index=False))
    print("\nData types:\n", df.dtypes.to_string())

    # 2. Data quality audit
    print("\n2. DATA QUALITY AUDIT")
    missing = pd.DataFrame({
        "Missing Values": df.isna().sum(),
        "Missing Percentage": (df.isna().mean() * 100).round(2)
    })
    print("\nMissing-value report:\n", missing[missing["Missing Values"] > 0].to_string())
    print("\nDuplicate rows:", df.duplicated().sum())

    # 3. Descriptive statistics
    print("\n3. DESCRIPTIVE STATISTICS")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().T
        stats["Median"] = df[numeric_cols].median()
        stats["Variance"] = df[numeric_cols].var()
        stats["IQR"] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
        stats = stats.round(2)
        print(stats.to_string())
        stats.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    # 4. Univariate analysis
    print("\n4. UNIVARIATE ANALYSIS")
    for col in ["Age", "Income", "Sales", "Profit"]:
        if col in df.columns:
            print(f"\n{col} summary:\n", df[col].describe().round(2).to_string())
            plt.figure(figsize=(7, 4))
            plt.hist(df[col].dropna(), bins=30)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"distribution_{col.lower()}.png", dpi=150)
            plt.close()

    for col in ["Region", "Product_Category", "Customer_Segment"]:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            print(f"\n{col} frequency:\n", counts.to_string())
            counts.to_csv(OUTPUT_DIR / f"frequency_{col.lower()}.csv")
            ax = counts.plot(kind="bar", figsize=(7, 4), title=f"Count of {col}")
            ax.set_ylabel("Number of transactions")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"count_{col.lower()}.png", dpi=150)
            plt.close()

    # 5. Outlier detection using IQR method
    print("\n5. OUTLIER DETECTION (IQR METHOD)")
    outlier_rows = []
    for col in ["Sales", "Profit", "Age", "Income"]:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((df[col] < lower) | (df[col] > upper)).sum())
            outlier_rows.append({"Column": col, "Lower Limit": lower,
                                 "Upper Limit": upper, "Outlier Count": count})
            print(f"{col}: lower={lower:.2f}, upper={upper:.2f}, outliers={count}")
            plt.figure(figsize=(7, 3))
            plt.boxplot(df[col].dropna(), vert=False)
            plt.title(f"Outlier Check - {col}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"outliers_{col.lower()}.png", dpi=150)
            plt.close()
    pd.DataFrame(outlier_rows).to_csv(OUTPUT_DIR / "outlier_summary.csv", index=False)

    # 6. Feature engineering (create columns if they are not already present)
    print("\n6. FEATURE ENGINEERING")
    if "Profit_Margin" not in df.columns and {"Profit", "Sales"}.issubset(df.columns):
        df["Profit_Margin"] = np.where(df["Sales"] != 0,
                                        (df["Profit"] / df["Sales"]) * 100, np.nan)
    if "Delivery_Days" not in df.columns and {"Order_Date", "Delivery_Date"}.issubset(df.columns):
        order = pd.to_datetime(df["Order_Date"], errors="coerce", dayfirst=True)
        delivery = pd.to_datetime(df["Delivery_Date"], errors="coerce", dayfirst=True)
        df["Delivery_Days"] = (delivery - order).dt.days
    if "Revenue_per_Customer" not in df.columns and {"Customer_ID", "Sales"}.issubset(df.columns):
        df["Revenue_per_Customer"] = df.groupby("Customer_ID")["Sales"].transform("sum")
    feature_cols = [c for c in ["Transaction_ID", "Profit_Margin", "Delivery_Days",
                                 "Revenue_per_Customer"] if c in df.columns]
    if feature_cols:
        print(df[feature_cols].head().to_string(index=False))

    # 7. Business questions
    print("\n7. BUSINESS QUESTIONS")
    business_tables = {}
    if {"Region", "Sales"}.issubset(df.columns):
        business_tables["region_total_sales"] = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    if {"Product_Category", "Profit"}.issubset(df.columns):
        business_tables["category_total_profit"] = df.groupby("Product_Category")["Profit"].sum().sort_values(ascending=False)
    if {"Customer_Segment", "Profit"}.issubset(df.columns):
        business_tables["segment_total_profit"] = df.groupby("Customer_Segment")["Profit"].sum().sort_values(ascending=False)
    if {"Payment_Method", "Sales"}.issubset(df.columns):
        business_tables["payment_method_sales"] = df.groupby("Payment_Method")["Sales"].sum().sort_values(ascending=False)
    if "Delivery_Mode" in df.columns:
        agg = {"Transaction_ID": "count"} if "Transaction_ID" in df.columns else {"Delivery_Mode": "count"}
        delivery_summary = df.groupby("Delivery_Mode").agg(Total_Orders=(next(iter(agg)), "count"))
        if "Delivery_Days" in df.columns:
            delivery_summary["Average_Delivery_Days"] = df.groupby("Delivery_Mode")["Delivery_Days"].mean()
        business_tables["delivery_mode_summary"] = delivery_summary.sort_values("Total_Orders", ascending=False)
    if {"Region", "Sales"}.issubset(df.columns):
        business_tables["region_average_sales"] = df.groupby("Region")["Sales"].mean().sort_values(ascending=False)

    for name, table in business_tables.items():
        print(f"\n{name.replace('_', ' ').title()}:\n", table.round(2).to_string())
        table.to_csv(OUTPUT_DIR / f"{name}.csv")

    # 8. Main comparison charts
    if {"Region", "Sales"}.issubset(df.columns):
        df.groupby("Region")["Sales"].sum().sort_values().plot(kind="bar", figsize=(8, 4), title="Total Sales by Region")
        plt.ylabel("Sales")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "total_sales_by_region.png", dpi=150)
        plt.close()
    if {"Product_Category", "Profit"}.issubset(df.columns):
        df.groupby("Product_Category")["Profit"].sum().sort_values().plot(kind="bar", figsize=(8, 4), title="Total Profit by Product Category")
        plt.ylabel("Profit")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "total_profit_by_category.png", dpi=150)
        plt.close()

    # Save the dataset with any feature columns created above.
    df.to_csv(OUTPUT_DIR / "retail_analysis_dataset.csv", index=False)
    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("All output tables and charts are saved in:", OUTPUT_DIR)
    print("=" * 65)


if __name__ == "__main__":
    main()

