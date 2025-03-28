# 引入FDI的滞后变量
data['FDI_lag'] = data.groupby('Country')['FDI'].shift(1)

# 固定效应回归，包含滞后变量
fixed_effect_lag_model = PanelOLS.from_formula('GWG ~ FDI_lag + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
fixed_effect_lag_result = fixed_effect_lag_model.fit()

# 输出结果
print(fixed_effect_lag_result.summary)
