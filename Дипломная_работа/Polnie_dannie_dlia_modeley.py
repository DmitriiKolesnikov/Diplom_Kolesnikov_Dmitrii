import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import statsmodels.api as sm

df = pd.read_csv("/Users/jimsgood/Дипломная_работа/DATA_FOR_LOG_REGRESSION_TRANSFORMED.csv")

df.dropna(inplace=True)
df['const'] = 1

total_obs = len(df)
quarters = np.arange(1, total_obs+1)
df['phase'] = pd.cut(quarters, bins=[0, 12, 20, 28, 36], labels=['Growth','Peak','Recession','Recovery'])

data_growth = df[df['phase']=='Growth']
data_peak = df[df['phase']=='Peak']
data_recession = df[df['phase']=='Recession']
data_recovery = df[df['phase']=='Recovery']

print("Defaults by phase:",
      data_growth['default'].sum(),
      data_peak['default'].sum(),
      data_recession['default'].sum(),
      data_recovery['default'].sum())

features = ['const','gdp_growth','inflation','key_rate','key_rate_sq',
            'log_N1','NPL','ROA','H3','log_assets','share_sys']

y_full = df['default']
X_full = df[features]
logit_full = sm.Logit(y_full, X_full).fit(disp=0)

logit_models = {}
for phase, data_phase in zip(['Peak','Recession','Recovery'], [data_peak, data_recession, data_recovery]):
    if data_phase['default'].sum() == 0:
        logit_models[phase] = None
        continue
    y = data_phase['default']
    X = data_phase[features]
    logit_models[phase] = sm.Logit(y, X).fit(disp=0)

def summarize_logit(result):
    params = result.params
    stderr = result.bse
    pvals = result.pvalues
    summary_table = []
    for idx in params.index:
        coef = params[idx]
        se = stderr[idx]
        p = pvals[idx]
        sig = ''
        if p < 0.01: sig = '***'
        elif p < 0.05: sig = '**'
        elif p < 0.10: sig = '*'
        summary_table.append((idx, coef, se, p, sig))
    return summary_table

print("Полные коэффициенты логистической регрессии")
for var, coef, se, p, sig in full_summary:
    print(f"{var}: {coef:.4f} (SE={se:.4f}, p={p:.3f}) {sig}")
for phase, model in logit_models.items():
    if model is None:
        print(f"\n{phase} нет ничего")
    else:
        phase_summary = summarize_logit(model)
        print(f"\n{phase} Коэффициенты по фазам логистической регрессии:")
        for var, coef, se, p, sig in phase_summary:
            print(f"{var}: {coef:.4f} (SE={se:.4f}, p={p:.3f}) {sig}")

def evaluate_models(X_train, X_test, y_train, y_test):
    log_clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_clf.fit(X_train, y_train)
    y_pred_log = log_clf.predict(X_test)
    y_proba_log = log_clf.predict_proba(X_test)[:,1]

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    y_proba_rf = rf_clf.predict_proba(X_test)[:,1]

    metrics = {}
    for model, y_pred, y_proba, name in [(y_pred_log, y_pred_log, y_proba_log, 'Logistic'),
                                         (y_pred_rf, y_pred_rf, y_proba_rf, 'RandomForest')]:
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
    return metrics

performance = {}
for phase, data_phase in zip(['Full','Peak','Recession','Recovery'],
                             [df, data_peak, data_recession, data_recovery]):
    X = data_phase[['gdp_growth','inflation','key_rate','key_rate_sq','log_N1','NPL','ROA','H3','log_assets','share_sys']]
    y = data_phase['default']
    if y.nunique() < 2:
        performance[phase] = None
        continue
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    performance[phase] = evaluate_models(X_train, X_test, y_train, y_test)

for phase, metrics in performance.items():
    if metrics is None:
        print(f"{phase}  нет ничего ")
    else:
        print(f"\n{phase} показатели фазы:")
        for model_name, vals in metrics.items():
            acc = vals['accuracy']*100; prec = vals['precision']*100
            rec = vals['recall']*100; f1 = vals['f1']*100; auc = vals['roc_auc']*100
            print(f" {model_name}: Accuracy={acc:.2f}%, Precision={prec:.1f}%, Recall={rec:.1f}%, F1={f1:.2f}%, ROC AUC={auc:.3f}%")
