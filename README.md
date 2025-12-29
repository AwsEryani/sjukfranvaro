## Sjukfrånvaro Prognostisering - Sickness Absence Forecasting App

This web application provides a user-friendly interface to analyze historical sickness absence data and generate future forecasts using a machine learning model.

## Overview

The project is a Flask-based web application that allows users to upload a CSV file containing sickness absence records. The backend processes this data, trains an XGBoost regression model, and generates predictions. The results, including a performance score (R²), a comparison plot, and a detailed data table, are then dynamically displayed on a modern, responsive web interface.

## Features

*   **Easy Data Upload:** Simple and intuitive interface to upload CSV files.
*   **Asynchronous Processing:** Analysis runs in a background thread, so the user interface remains responsive.
*   **Live Progress Tracking:** A real-time progress bar shows the status of the analysis, from data loading to model training.
*   **Dynamic Model Training:** An XGBoost model is trained from scratch on your specific data with every upload.
*   **Insightful Visualizations:** A plot comparing actual vs. predicted sickness absence rates is generated for the most recent year in the dataset.
*   **Detailed Results:** A paginated table provides a month-by-month breakdown of the model's predictions.
*   **Performance Metrics:** The model's accuracy is measured using the R² (R-squared) score, giving you a clear indication of its performance.
*   **Modern UI:** A responsive, dark-themed interface built with Bootstrap 5.

## How The Process Works

The application follows a clear, step-by-step process from data upload to result visualization.

1.  **Step 1: Upload Data**
    The user selects a CSV file via the web form and clicks "Ladda upp och analysera".

2.  **Step 2: Initiate Analysis**
    The Flask backend receives the file. It doesn't start the heavy processing immediately. Instead, it:
    *   Saves the uploaded file to the server with a unique, timestamp-based name (e.g., `20251001_143000.csv`).
    *   Creates a unique **Task ID** for this specific analysis job.
    *   Starts the entire analysis process in a **background thread**.
    *   Immediately sends the **Task ID** back to the user's browser.

3.  **Step 3: Track Progress**
    The JavaScript on the frontend receives the Task ID and begins polling a progress endpoint (e.g., `/progress/<task_id>`) every few seconds. The backend responds with the current status (`'Loading data'`, `'Training model'`, etc.) and a percentage, which is used to update the progress bar on the screen.

4.  **Step 4: Data Processing & Model Training (In the Background)**
    This is where the core data science work happens:
    *   **Load & Preprocess:** The data from the CSV is loaded, cleaned, and transformed. Features like dates, age groups, and job positions are converted into a numerical format suitable for the model.
    *   **Split Data:** The script automatically identifies the **most recent year** in the dataset and sets it aside as the **test set**. All data from previous years is designated as the **training set**.
    *   **Train Model:** A new XGBoost machine learning model is trained exclusively on the **training set**.

5.  **Step 5: Prediction & Evaluation**
    *   The newly trained model is used to predict the sickness absence rates for the **test set** (the most recent year).
    *   The model's predictions are compared against the actual, known values from the test set to calculate the **R² score**, which measures how well the model performed.

6.  **Step 6: Display Results**
    *   Once the background task is complete, the frontend polling detects the 'Complete' status.
    *   It makes one final request to a result endpoint (e.g., `/result/<task_id>`).
    *   The backend sends back the final results: the plot image, the HTML for the data table, and the R² score.
    *   JavaScript dynamically injects this content into the page, displaying the complete analysis to the user without requiring a page reload.

## How Are Data Updates Handled?

The model is **stateless**, meaning it does not remember or build upon previous uploads. The analysis is performed exclusively on the data contained within the single file you upload at that moment.

**If your data gets updated (e.g., you have new data for recent months), you must follow this process:**

1.  Update your master CSV file to include the new rows of data.
2.  Upload this **complete, updated CSV file** to the application.

The application will then repeat the entire training and prediction cycle from scratch using this new file. This ensures that the model is always trained on the most current historical data you provide, and its predictions reflect the latest trends and patterns present in your dataset.


## Performance Interpretation

The validation error steadily improves from 0.06245 → 0.02202, which is about a 65% reduction in RMSE over training.
The app also reports R² after prediction, which is another useful performance indicator (closer to 1.0 is better). Use RMSE + R² together to judge quality.

------------------------------------------------------------------------------
How to Run the Project
------------------------------------------------------------------------------
git pull repo

# Create a virtual environment & Activate it
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. cd health_project
5. python app.py
6. Running on http://127.0.0.1:5002
7. Upload data/synthetic_dataset.csv to test
