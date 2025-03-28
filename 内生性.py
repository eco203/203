from linearmodels.panel import PanelOLS

# 第一阶段：回归FDI与工具变量HIST_TRADE
first_stage = PanelOLS.from_formula('FDI ~ HIST_TRADE + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
first_stage_result = first_stage.fit()

# 输出第一阶段回归结果
print(first_stage_result.summary)

# 使用第一阶段的预测结果作为新的FDI
data['predicted_FDI'] = first_stage_result.fitted_values

# 第二阶段：回归性别工资差距（GWG）与预测的FDI
second_stage = PanelOLS.from_formula('GWG ~ predicted_FDI + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
second_stage_result = second_stage.fit()

# 输出第二阶段回归结果
print(second_stage_result.summary)
