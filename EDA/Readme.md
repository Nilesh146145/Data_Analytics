#  Exploratory Data Analysis (EDA) on Retail Sales Dataset

##  Project Description

This project performs **Exploratory Data Analysis (EDA)** on a retail sales dataset. The goal is to uncover insights, patterns, and trends that can help businesses make better decisions. The dataset contains customer transactions, product categories, sales amounts, and demographic information.

---

##  Dataset Overview

- **Rows:** 1000
- **Columns:** 9
- **Fields:**
  - `Transaction ID`
  - `Date`
  - `Customer ID`
  - `Gender`
  - `Age`
  - `Product Category`
  - `Quantity`
  - `Price per Unit`
  - `Total Amount`

Dataset Source: [Kaggle - Retail Sales Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)

---

## 🔍 Objectives

- Load and clean the dataset
- Analyze basic statistics (mean, median, mode, std)
- Perform time-series analysis on sales trends
- Analyze customer demographics and purchasing behavior
- Visualize key insights using bar plots, line charts, and heatmaps
- Provide actionable recommendations

---

##  Technologies Used

- **Python**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**

---

##  Key Findings

- Total sales trend over time visualized using time-series plots
- Sales analysis by:
  - Gender
  - Product Category
  - Age group
- Correlation analysis between quantity, price, and total amount
- Statistical overview of numeric columns

---

##  Sample Insights

- **Top Selling Category:** Electronics
- **High Purchasing Gender:** Female
- **Customer Age Range:** 18–64
- **Strong Correlation:** Quantity × Price per Unit = Total Amount

---

##  Recommendations

1. Focus on high-performing product categories like **Electronics**.
2. Launch targeted promotions based on **gender** and **age** segments.
3. Explore seasonal trends to adjust **inventory and marketing** efforts.

---

## 🗂 Project Structure

```bash
├── RetailSalesDataset.csv
├── EDA_Retail_Sales.ipynb       # Jupyter Notebook with complete EDA
├── Report.pdf                   # (Optional) PDF report of findings
├── README.md                    # Project overview and documentation
