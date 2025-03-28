from linearmodels.panel import PanelOLS

# 第一阶段：回归 FDI 对女性就业比例 FEM_EMP 的影响
first_stage_model = PanelOLS.from_formula('FEM_EMP ~ FDI + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
first_stage_result = first_stage_model.fit()

# 输出第一阶段回归结果
print("第一阶段回归结果：")
print(first_stage_result.summary)

# 第二阶段：回归 FEM_EMP 对性别工资差距（GWG）的影响
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
