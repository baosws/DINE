from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
import pandas as pd

def F1(A_true, A_pred):
    return f1_score(A_true.ravel(), A_pred.ravel())

def Precision(A_true, A_pred):
    return precision_score(A_true.ravel(), A_pred.ravel())

def Recall(A_true, A_pred):
    return recall_score(A_true.ravel(), A_pred.ravel())

def scores(df):
    y_pred = df['Pred']
    y_true = df['GT']
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    res = {
        'MAE': mae,
        'RMSE': rmse,
        'Time': df['Time'].mean()
    }
    return pd.Series(res)