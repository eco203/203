import pandas as pd
import numpy as np

# 设置随机种子，以确保结果可复现
np.random.seed(42)

# 模拟20个国家（编号1到20）和25年（2000到2024）
countries = [f'Country_{i}' for i in range(1, 21)]
years = list(range(2000, 2025))

# 模拟数据
n_countries = len(countries)
n_years = len(years)

# 创建面板数据框架
data = pd.DataFrame({
    'Country': np.random.choice(countries, n_countries * n_years, replace=True),
    'Year': np.random.choice(years, n_countries * n_years, replace=True),
    'FDI': np.random.uniform(0, 5, n_countries * n_years),  # FDI占GDP的比重
    'GDP_per_capita': np.random.uniform(10000, 50000, n_countries * n_years),  # 人均GDP
    'Total_average_wage': np.random.uniform(1000, 5000, n_countries * n_years),  # 总平均工资
    'HDI': np.random.uniform(0.7, 1.0, n_countries * n_years),  # 人类发展指数
    'Fertility': np.random.uniform(1.2, 3.5, n_countries * n_years),  # 生育率
    'GWG': np.random.uniform(0, 0.5, n_countries * n_years),  # 性别工资差距
    'FEM_EMP': np.random.uniform(0, 1, n_countries * n_years), # 女性就业比例
    'HIST_TRADE': np.random.uniform(0, 1, n_countries * n_years)  # 历史贸易关系指数
})

# 将 'Country' 和 'Year' 设置为索引，模拟面板数据结构
data.set_index(['Country', 'Year'], inplace=True)

# 查看前几行数据
print(data.head())

from linearmodels.panel import PanelOLS
first_stage_model = PanelOLS.from_formula('FEM_EMP ~ FDI + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
first_stage_result = first_stage_model.fit()

# 输出第一阶段回归结果
print("第一阶段回归结果：")
print(first_stage_result.summary)

# 第二阶段：女性就业比例对性别工资差距的影响
second_stage_model = PanelOLS.from_formula('GWG ~ FEM_EMP + FDI + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
second_stage_result = second_stage_model.fit()

# 输出第二阶段回归结果
print("第二阶段回归结果：")
print(second_stage_result.summary)

# 获取第一阶段和第二阶段的系数
beta_1 = first_stage_result.params['FDI']
beta_2 = second_stage_result.params['FEM_EMP']

# 计算间接效应
indirect_effect = beta_1 * beta_2
print(f"间接效应 (FDI 通过 FEM_EMP 对 GWG 的影响): {indirect_effect:.4f}")

# 获取第二阶段回归中 FDI 对 GWG 的系数（直接效应）
direct_effect = second_stage_result.params['FDI']

# 计算总效应
total_effect = direct_effect + indirect_effect

print(f"直接效应 (FDI 对 GWG 的影响): {direct_effect:.4f}")
print(f"总效应 (Direct Effect + Indirect Effect): {total_effect:.4f}")
from linearmodels.panel import PanelOLS

# 第一阶段：回归FDI与工具变量HIST_TRADE

# 输出第一阶段回归结果
print(first_stage_result.summary)

# 使用第一阶段的预测结果作为新的FDI
data['predicted_FDI'] = first_stage_result.fitted_values

# 第二阶段：回归性别工资差距（GWG）与预测的FDI
second_stage = PanelOLS.from_formula('GWG ~ predicted_FDI + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
second_stage_result = second_stage.fit()

# 输出第二阶段回归结果
print(second_stage_result.summary)

from linearmodels.panel import PanelOLS

# 固定效应回归模型，包含FDI、GDP、人均收入等变量
fixed_effect_model = PanelOLS.from_formula('GWG ~ FDI + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)

# 拟合模型
fixed_effect_result = fixed_effect_model.fit()

# 输出回归结果
print(fixed_effect_result.summary)

from linearmodels.panel import RandomEffects

# 随机效应模型：注意这里不在公式中加入 "RandomEffects"
random_effect_model = RandomEffects.from_formula('GWG ~ FDI + GDP_per_capita + Total_average_wage + HDI + Fertility', data)
random_effect_result = random_effect_model.fit()

# Hausman检验：比较固定效应模型和随机效应模型
from linearmodels.panel import compare
results = compare({'Fixed Effects': fixed_effect_result, 'Random Effects': random_effect_result})
print(results)

# 引入FDI的滞后变量
data['FDI_lag'] = data.groupby('Country')['FDI'].shift(1)

# 固定效应回归，包含滞后变量
fixed_effect_lag_model = PanelOLS.from_formula('GWG ~ FDI_lag + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
fixed_effect_lag_result = fixed_effect_lag_model.fit()

# 输出结果
print(fixed_effect_lag_result.summary)

# 替换FDI流入为FDI存量，生成FDI存量数据（此处仅为示例，实际情况应根据数据计算FDI存量）
data['FDI_stock'] = data['FDI'] * np.random.uniform(0.5, 1.5, size=data.shape[0])

# 使用FDI存量进行回归
fixed_effect_fdi_stock_model = PanelOLS.from_formula('GWG ~ FDI_stock + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
fixed_effect_fdi_stock_result = fixed_effect_fdi_stock_model.fit()

# 输出结果
print(fixed_effect_fdi_stock_result.summary)






