# Retention Risk Analysis – Customer Churn Prediction System

## Project Logbook – Sem Project

| Sr.No | Contents | Date |
|-------|----------|-------------|
| 1 | Project Group Formation | 08/01/2026 – 15/01/2026 |
| 2 | Project Topic Finalization | 19/01/2026 – 29/01/2026 |
| 3 | Identified and analyzed the functional and non-functional requirements of the proposed system | 29/01/2026 – 31/01/2026 |
| 4 | Studied customer retention, churn factors, and analyzed the overall system workflow | 07/02/2026 – 18/02/2026 |
| 5 | Implementation Phase – I | 19/02/2026 – 28/02/2026 |
| 6 | Collected, cleaned, and preprocessed the customer churn dataset | 01/03/2026 – 09/03/2026 |
| 7 | Implementation Phase – II: Developed and trained the machine learning model | 10/03/2026 – 27/03/2026 |
| 8 | Implementation Phase – III: Developed the web interface and integrated the prediction model | 28/03/2026 – 01/04/2026 |
| 9 | Conducted functional, performance, and validation testing of the complete system | 05/04/2026 – 27/04/2026 |
| 10 | Evaluated model performance using relevant classification metrics and analyzed the results | 28/04/2026 – 09/05/2026 |
| 11 | Prepared the project documentation and summarized the conclusions and future scope | 09/05/2026 – 11/05/2026 |

---

# Retention Risk Analysis – Customer Churn Prediction System

## Overview

The **Retention Risk Analysis – Customer Churn Prediction System** is a machine learning-based web application designed to predict whether a customer is likely to leave a service or company.

The system analyzes customer information such as tenure, monthly charges, contract type, payment method, internet service, and other relevant attributes. Based on these factors, a machine learning classification model predicts whether the customer is at **high risk of churn or likely to be retained**.

The project aims to help organizations identify customers who may leave and take appropriate retention measures in advance. This can help reduce customer loss, improve customer satisfaction, and support data-driven business decisions.

---

## Features

- Customer Data Input
- Customer Churn Prediction
- Retention Risk Identification
- Machine Learning-Based Classification
- Churn Probability / Confidence Score
- Customer Risk Level Display
- User-Friendly Web Interface
- Automated Prediction Process
- Data-Driven Decision Support
- Responsive Web Design

---

## Project Architecture

```text
        User Enters Customer Details
                    │
                    ▼
          Data Preprocessing
                    │
                    ▼
        Feature Transformation
                    │
                    ▼
       Machine Learning Model
                    │
                    ▼
       Churn Probability Prediction
                    │
                    ▼
       Risk Classification
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       Low Risk            High Risk
       / Retained           / Churn
          │                   │
          └─────────┬─────────┘
                    ▼
             Display Result
                    │
                    ▼
          Retention Decision
