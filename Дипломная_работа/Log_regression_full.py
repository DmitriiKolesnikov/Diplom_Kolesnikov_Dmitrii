import pandas as pd

file_path = 'ДАТА_МОДЕЛИ.xlsx'
df_raw = pd.read_excel(file_path, sheet_name='Лог регрессия', header=None)

idx_years = df_raw[df_raw.apply(lambda row: row.str.contains('Год', na=False).any(), axis=1)].index[0]
idx_quarters = df_raw[df_raw.apply(lambda row: row.str.contains('Квартал', na=False).any(), axis=1)].index[0]
header_years = df_raw.iloc[idx_years].ffill()
header_quarters = df_raw.iloc[idx_quarters]

idx_inflation = df_raw[df_raw.iloc[:,1] == 'Инфляция'].index[0]
idx_gdp_growth = df_raw[df_raw.iloc[:,1].str.contains('Темп прироста ВВП', na=False)].index[0]
idx_key_rate = df_raw[df_raw.iloc[:,1].str.contains('Ключевая ставка', na=False)].index[0]

records_macro = []
for col in range(2, df_raw.shape[1]):
    year = header_years[col]
    quarter = header_quarters[col]
    if pd.isna(year) or not isinstance(quarter, str):
        continue
    try:
        year = int(year)
    except:
        continue
    records_macro.append({
        'year': year,
        'quarter': quarter,
        'gdp_growth': df_raw.iloc[idx_gdp_growth, col],
        'inflation': df_raw.iloc[idx_inflation, col],
        'key_rate': df_raw.iloc[idx_key_rate, col]
    })
df_macro = pd.DataFrame(records_macro)

vars_bank = [
    'Н1', 'NPL', 'ROA', 'H3', 'log_assets', 'share_sys',
    'g_credit', 'LDR', 'interbank', 'g_profit', 'ownership', 'default'
]
start_bank = idx_key_rate + 1
nvars = len(vars_bank)
records_banks = []

for i in range(start_bank, df_raw.shape[0] - nvars + 1, nvars):
    block = df_raw.iloc[i:i+nvars]
    bank_name = block.iloc[0, 0]
    if pd.isna(bank_name):
        continue
    for j, var in enumerate(vars_bank):
        for col in range(2, df_raw.shape[1]):
            year = header_years[col]
            quarter = header_quarters[col]
            if pd.isna(year) or not isinstance(quarter, str):
                continue
            try:
                year = int(year)
            except:
                continue
            records_banks.append({
                'bank': bank_name,
                'year': year,
                'quarter': quarter,
                var: block.iloc[j, col]
            })

df_banks = pd.DataFrame(records_banks)
df_banks = df_banks.pivot_table(
    index=['bank', 'year', 'quarter'],
    values=vars_bank,
    aggfunc='first'
).reset_index()

df_merged = pd.merge(df_banks, df_macro, on=['year', 'quarter'], how='left')

target = 'default'
features = [
    'gdp_growth',
    'inflation',
    'key_rate',
    'Н1',
    'NPL',
    'ROA',
    'H3',
    'log_assets',
    'share_sys'
]

df_lr_model = df_merged[['bank', 'year', 'quarter', target] + features].copy()

df_lr_model = df_lr_model[df_lr_model['year'] != 2002]
banks_def_2009 = df_lr_model[(df_lr_model['year']==2009) & (df_lr_model['default']==1)]['bank'].unique()
df_lr_model.loc[(df_lr_model['bank'].isin(banks_def_2009)) & (df_lr_model['year']==2008), 'default'] = 1
df_lr_model = df_lr_model[~((df_lr_model['bank'].isin(banks_def_2009)) & (df_lr_model['year']>=2009))]

df_lr_model[target] = df_lr_model[target].astype(int)
for feat in features:
    df_lr_model[feat] = pd.to_numeric(df_lr_model[feat], errors='coerce')

df_lr_model = df_lr_model.drop(columns=['bank', 'year', 'quarter'])

output_filename = 'DATA_FOR_LOG_REGRESSION_FULL.csv'
df_lr_model.to_csv(output_filename, index=False)

print("Файл для обучения логистической регрессии сохранён:", output_filename)
