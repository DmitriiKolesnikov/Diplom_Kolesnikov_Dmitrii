import pandas as pd
import numpy as np

df = pd.read_csv('/Users/jimsgood/Дипломная_работа/DATA_FOR_LOG_REGRESSION_FULL.csv')

df['key_rate_sq'] = df['key_rate'] ** 2
df['log_N1'] = np.log(df['Н1'])

features = [
    'gdp_growth', 'inflation',
    'key_rate', 'key_rate_sq',
    'log_N1', 'NPL',
    'ROA', 'H3',
    'log_assets', 'share_sys'
]

df_transformed = df[features + ['default']]
df_transformed.to_csv('DATA_FOR_LOG_REGRESSION_TRANSFORMED.csv', index=False)
