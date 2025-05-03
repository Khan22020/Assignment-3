# 📊 Financial Data ML App using Streamlit

AF3005 – Programming for Finance  
FAST National University of Computer and Emerging Sciences (FAST-NUCES), Islamabad  
👨‍🏫 Instructor: Dr. Usama Arshad  
🎓 Program: BS Financial Technology (BSFT) – Spring 2025  
📚 Sections: BSFT06A, BSFT06B, BSFT06C

---

## 🔍 Project Overview

This is a fully interactive Streamlit app that helps users perform **Machine Learning** on **financial data**. You can either upload a `.csv` file or fetch **real-time stock data from Yahoo Finance**. The app guides you through a **step-by-step ML pipeline** using one of the following models:

- Linear Regression
- Logistic Regression
- K-Means Clustering

Everything is done visually — no coding required for interaction!

---

## 🎯 Purpose of This Assignment

The goal is to **build practical ML skills** in finance by:

- Understanding and applying ML concepts step-by-step
- Working with real financial data (from CSV files and Yahoo Finance)
- Creating a user-friendly dashboard with charts, animations, and themes
- Learning to visualize and evaluate models in an interactive way

---

## 🧠 Learning Outcomes

- Build ML models for real-world financial decision-making
- Interpret and visualize data effectively using Python tools
- Improve self-learning and development skills in Python for finance

---

## 🚀 What This App Can Do

✅ Upload a CSV file or search for any stock using Yahoo Finance  
✅ Choose your model (Linear, Logistic, or K-Means)  
✅ Go through each ML step via buttons with success/info messages  
✅ See clean visualizations at each step  
✅ Get final predictions/clusters, with interactive charts  
✅ Bonus: Download the results!

---

## 🛠️ ML Pipeline Steps

Each stage is separated by buttons for better control and understanding:

| Step               | What It Does                                            | Output                                           |
|--------------------|----------------------------------------------------------|--------------------------------------------------|
| **1. Load Data**    | Upload CSV or fetch stock data using a ticker symbol    | Data table + success message                    |
| **2. Preprocess**   | Handle missing values, clean the data                   | Info message + null values table                |
| **3. Engineer Features** | Select or transform features for modeling            | Display selected features or transformations    |
| **4. Split Data**   | Split into training and testing sets                    | Visualized using pie or bar chart               |
| **5. Train Model**  | Fit the selected model to training data                 | Training complete notification                  |
| **6. Evaluate Model** | Show how well the model performed                      | Accuracy, precision, or clustering visuals      |
| **7. Show Results** | Final predictions or cluster outputs                    | Graphs, tables, and download option             |

---

## 💻 How to Use the App

### 🔧 Setup Instructions

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/financial-ml-app.git
   cd financial-ml-app
