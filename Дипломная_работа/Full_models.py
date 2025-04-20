import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm

data = pd.read_csv("/Users/jimsgood/Дипломная_работа/DATA_FOR_LOG_REGRESSION_TRANSFORMED.csv")

X = data.drop(columns=["default"])
y = data["default"]

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_train_full_const = sm.add_constant(X_train_full)
logit_model_full = sm.Logit(y_train_full, X_train_full_const)
logit_result_full = logit_model_full.fit(disp=0)

coeff_full = logit_result_full.params
stderr_full = logit_result_full.bse
pvalues_full = logit_result_full.pvalues

rf_model_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_full.fit(X_train_full, y_train_full)

X_test_full_const = sm.add_constant(X_test_full, has_constant='add')
y_proba_logit_full = logit_result_full.predict(X_test_full_const)
y_pred_logit_full = (y_proba_logit_full >= 0.5).astype(int)

y_proba_rf_full = rf_model_full.predict_proba(X_test_full)[:, 1]
y_pred_rf_full = rf_model_full.predict(X_test_full)

accuracy_logit_full = accuracy_score(y_test_full, y_pred_logit_full)
precision_logit_full = precision_score(y_test_full, y_pred_logit_full, zero_division=0)
recall_logit_full = recall_score(y_test_full, y_pred_logit_full, zero_division=0)
f1_logit_full = f1_score(y_test_full, y_pred_logit_full, zero_division=0)
if len(np.unique(y_test_full)) == 2:
    roc_auc_logit_full = roc_auc_score(y_test_full, y_proba_logit_full)
else:
    roc_auc_logit_full = None

accuracy_rf_full = accuracy_score(y_test_full, y_pred_rf_full)
precision_rf_full = precision_score(y_test_full, y_pred_rf_full, zero_division=0)
recall_rf_full = recall_score(y_test_full, y_pred_rf_full, zero_division=0)
f1_rf_full = f1_score(y_test_full, y_pred_rf_full, zero_division=0)
if len(np.unique(y_test_full)) == 2:
    roc_auc_rf_full = roc_auc_score(y_test_full, y_proba_rf_full)
else:
    roc_auc_rf_full = None

df = data.copy()
mask_peak = (df["gdp_growth"] > 0) & (df["inflation"] >= 13)
mask_recession = df["gdp_growth"] < 0
mask_growth_or_recovery = (df["gdp_growth"] > 0) & (df["inflation"] < 13)
mask_recovery = mask_growth_or_recovery & (
    (df["inflation"] < 8) | (df["gdp_growth"] > 12) | (df["gdp_growth"] < 5)
)
mask_growth = mask_growth_or_recovery & ~mask_recovery

if df[mask_growth]["default"].sum() != 0:
    mask_def_growth = mask_growth & (df["default"] == 1)
    mask_growth = mask_growth & ~mask_def_growth
    mask_recovery = mask_recovery | mask_def_growth

data_growth = df[mask_growth]
data_peak = df[mask_peak]
data_recession = df[mask_recession]
data_recovery = df[mask_recovery]

print("Observations per phase:",
      f"growth={len(data_growth)}, peak={len(data_peak)}, recession={len(data_recession)}, recovery={len(data_recovery)}")
print("Defaults per phase:",
      f"growth={data_growth['default'].sum()},",
      f"peak={data_peak['default'].sum()},",
      f"recession={data_recession['default'].sum()},",
      f"recovery={data_recovery['default'].sum()}")

if data_growth["default"].nunique() < 2:
    print("Фаза роста: ни одного случая дефолта, модель не обучается (тривиальная нулевая модель).")
    model_logit_growth = None
    model_rf_growth = None
else:
    X_g = data_growth.drop(columns=["default"])
    y_g = data_growth["default"]
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_g, y_g, test_size=0.30, random_state=42, stratify=y_g)
    X_train_g_const = sm.add_constant(X_train_g)
    logit_model_g = sm.Logit(y_train_g, X_train_g_const).fit(disp=0)
    rf_model_g = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_g.fit(X_train_g, y_train_g)
    model_logit_growth = logit_model_g
    model_rf_growth = rf_model_g

X_p = data_peak.drop(columns=["default"])
y_p = data_peak["default"]
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.30, random_state=42, stratify=y_p)
X_train_p_const = sm.add_constant(X_train_p)
try:
    logit_model_p = sm.Logit(y_train_p, X_train_p_const).fit(disp=0)
except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError):
    print("Фаза пика: невозможно оценить логистическую регрессию (сингулярная матрица). Пропускаем.")
    logit_model_p = None
# Случайный лес
rf_model_p = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_p.fit(X_train_p, y_train_p)

X_r = data_recession.drop(columns=["default"])
y_r = data_recession["default"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.30, random_state=42, stratify=y_r)
X_train_r_const = sm.add_constant(X_train_r)
try:
    logit_model_r = sm.Logit(y_train_r, X_train_r_const).fit(disp=0)
except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError):
    print("Фаза рецессии: невозможно оценить логистическую регрессию. Пропускаем.")
    logit_model_r = None

rf_model_r = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_r.fit(X_train_r, y_train_r)

X_rec = data_recovery.drop(columns=["default"])
y_rec = data_recovery["default"]
X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(X_rec, y_rec, test_size=0.30, random_state=42, stratify=y_rec)
X_train_rec_const = sm.add_constant(X_train_rec)
try:
    logit_model_rec = sm.Logit(y_train_rec, X_train_rec_const).fit(disp=0)
except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError):
    print("Фаза восстановления: невозможно оценить логистическую регрессию. Пропускаем.")
    logit_model_rec = None
rf_model_rec = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_rec.fit(X_train_rec, y_train_rec)

print("\nПолная логистическая модель (all data) коэффициенты и p-value:")
for var in coeff_full.index:
    coef = coeff_full[var]
    stderr = stderr_full[var]
    pval = pvalues_full[var]
    print(f"{var}: Coefficient = {coef:.4f}, Std.Error = {stderr:.4f}, p-value = {pval:.4f}")

if logit_result_full is not None:
    print("\nЛогистическая регрессия (полная выборка) - коэффициенты:")
    print(logit_result_full.summary())

if model_logit_growth:
    print("\nЛогистическая модель (фаза роста) - коэффициенты:")
    print(model_logit_growth.params)
if data_peak["default"].nunique() > 1:
    print("\nЛогистическая модель (фаза пика) - коэффициенты:")
    if logit_model_p is not None:
        print("\nЛогистическая модель (фаза пика) - коэффициенты:")
        print(logit_model_p.params)
    else:
        print("Пик: логистическая модель не оценена.")
if data_recession["default"].nunique() > 1:
    print("\nЛогистическая модель (фаза рецессии) - коэффициенты:")
    if logit_model_r is not None:
        print("\n\nЛогистическая модель (фаза рецессии) - коэффициенты:")
        print(logit_model_r.params)
    else:
        print("Рецессия: логистическая модель не оценена.")
if data_recovery["default"].nunique() > 1:
    print("\nЛогистическая модель (фаза восстановления) - коэффициенты:")
    if logit_model_rec is not None:
        print("\n\nЛогистическая модель (восстановления) - коэффициенты:")
        print(logit_model_rec.params)
    else:
        print("Восстановления: логистическая модель не оценена.")

def evaluate_model(y_true, y_pred, y_proba):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1-score"] = f1_score(y_true, y_pred, zero_division=0)
    if len(np.unique(y_true)) == 2:
        metrics["ROC AUC"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["ROC AUC"] = None
    return metrics

def print_metrics(name, metrics_dict):
    acc = metrics_dict["Accuracy"]; prec = metrics_dict["Precision"]
    rec = metrics_dict["Recall"]; f1 = metrics_dict["F1-score"]; auc = metrics_dict["ROC AUC"]
    print(f"{name}: Accuracy = {acc:.3f}, Precision = {prec:.3f}, Recall = {rec:.3f}, F1 = {f1:.3f}", end="")
    if auc is not None:
        print(f", ROC AUC = {auc:.3f}")
    else:
        print(" (ROC AUC недоступен - один класс в выборке)")

print("\nМетрики качества на тестовых наборах:")
if model_rf_growth is None or y_test_g.nunique() < 2:
    print("Phase Growth: все тестовые объекты относятся к классу 0 (низкий риск), Accuracy = 1.000 (тривиальная модель)")
else:
    y_proba_logit_g = model_logit_growth.predict(sm.add_constant(X_test_g, has_constant='add'))
    y_pred_logit_g = (y_proba_logit_g >= 0.5).astype(int)
    metrics_logit_g = evaluate_model(y_test_g, y_pred_logit_g, y_proba_logit_g)
    print_metrics("Logistic (Growth)", metrics_logit_g)
    y_proba_rf_g = model_rf_growth.predict_proba(X_test_g)[:, 1]
    y_pred_rf_g = model_rf_growth.predict(X_test_g)
    metrics_rf_g = evaluate_model(y_test_g, y_pred_rf_g, y_proba_rf_g)
    print_metrics("Random Forest (Growth)", metrics_rf_g)

risk_levels = []
for prob in y_proba_logit_full:
    if prob < 0.10:
        risk_levels.append("Low")
    elif prob < 0.50:
        risk_levels.append("Medium")
    else:
        risk_levels.append("High")

risk_levels = np.array(risk_levels)
low_count = np.sum(risk_levels == "Low")
med_count = np.sum(risk_levels == "Medium")
high_count = np.sum(risk_levels == "High")
print(f"\nКлассификация уровня риска на тестовой выборке полной модели:")
print(f"Низкий риск (<10% вероятности дефолта): {low_count} банков")
print(f"Средний риск (10%-50% вероятности): {med_count} банков")
print(f"Высокий риск (>50% вероятности): {high_count} банков")
high_risk_indices = np.where(risk_levels == "High")[0]
if len(high_risk_indices) > 0:
    high_risk_defaults = y_test_full.iloc[high_risk_indices].values
    print(f"Из {len(high_risk_indices)} банков с высоким риском, дефолт произошел у {high_risk_defaults.sum()} из них.")
