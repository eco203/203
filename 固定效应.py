from linearmodels.panel import PanelOLS

# 固定效应回归模型，包含FDI、GDP、人均收入等变量
fixed_effect_model = PanelOLS.from_formula('GWG ~ FDI + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)

# 拟合模型
fixed_effect_result = fixed_effect_model.fit()

# 输出回归结果
print(fixed_effect_result.summary)