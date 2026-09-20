# Telco Customer Churn Prediction

End-to-end machine learning solution for predicting customer churn at a telecommunications company.

## Artifact
https://drive.google.com/drive/folders/1uAKMGpHjFpgjrFVYHtXiRzWjjtYdiMWR?usp=drive_link

### Video Link
https://drive.google.com/file/d/1BXu17p4qCRy66yuevPqcRQl3uXJq91Nr/view?usp=sharing

### Repo Link
https://github.com/KumVivekAI/customer_churn

## Workflow Overview

This project follows the full assignment workflow:

**Business problem → Data loading → Preparation → EDA → Feature engineering → Decision Tree modelling → Evaluation → Interpretation → Saved pipeline → REST API**

The Jupyter notebook contains the analysis narrative, visualizations, and model interpretation. The `train_model.py` script retrains and saves the production pipeline used by the API.

## Project Structure

```
customer_churn_project/
├── data/
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   └── churn_model.pkl
├── src/
│   └── preprocessing.py
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── sample_request.json
```

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. From the project root, open the notebook or run training (imports expect `notebook/` as the working directory when using Jupyter, or run scripts from the project root).

## Train the Model

```bash
python train_model.py
```

This will:
- clean and preprocess the Telco dataset
- compare two decision tree configurations
- save the best model to `model/churn_model.pkl`
- save evaluation metrics to `model/model_metrics.json`

## Run the Notebook

```bash
jupyter notebook notebook/churn_analysis.ipynb
```

Run all cells from the `notebook/` folder (or use **Kernel → Run All**). The notebook includes data understanding, EDA, feature engineering, model comparison, evaluation, and interpretation with saved outputs after execution.

To execute the notebook from the command line (from project root):

```bash
python -m nbconvert --to notebook --execute --inplace notebook/churn_analysis.ipynb
```

## Start the API

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Interactive docs: `http://127.0.0.1:8000/docs`

## Sample API Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Sample Response

Using the provided `sample_request.json` (short-tenure, month-to-month contract):

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8028
}
```

Exact probability may vary slightly if the model is retrained; run `train_model.py` before calling the API.