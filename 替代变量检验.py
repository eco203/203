# 替换FDI流入为FDI存量，生成FDI存量数据（此处仅为示例，实际情况应根据数据计算FDI存量）
data['FDI_stock'] = data['FDI'] * np.random.uniform(0.5, 1.5, size=data.shape[0])

# 使用FDI存量进行回归
fixed_effect_fdi_stock_model = PanelOLS.from_formula('GWG ~ FDI_stock + GDP_per_capita + Total_average_wage + HDI + Fertility + EntityEffects', data)
fixed_effect_fdi_stock_result = fixed_effect_fdi_stock_model.fit()

# 输出结果
print(fixed_effect_fdi_stock_result.summary)
