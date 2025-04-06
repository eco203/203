import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 1. 导入数据 - 修改数据读取方式，考虑到第一行为国家，第一列为时间
def read_data_with_structure(file_path):
    # 读取Excel文件，不指定header和index_col
    df = pd.read_excel(file_path, header=None)
    # 获取国家列表（第一行，从第二列开始）
    countries = df.iloc[0, 1:].tolist()
    # 获取年份列表（第一列，从第二行开始）
    years = df.iloc[1:, 0].tolist()

    # 创建新的DataFrame来存储重组后的数据
    data_list = []

    # 遍历每个国家和年份，提取对应的数据
    for i, country in enumerate(countries):
        for j, year in enumerate(years):
            # i+1和j+1是因为我们跳过了第一行和第一列
            value = df.iloc[j + 1, i + 1]
            data_list.append({'Country': country, 'Year': year, 'Value': value})

    # 创建新的DataFrame
    new_df = pd.DataFrame(data_list)

    # 返回重组后的DataFrame
    return new_df


# 读取并重组所有数据文件
fdi_df = read_data_with_structure(r"D:\FDI_inflow_of_GDP_(%)(1).xlsx")
fdi_df.rename(columns={'Value': 'FDI_inflow'}, inplace=True)
fdi_df['FDI_inflow'] = pd.to_numeric(fdi_df['FDI_inflow'], errors='coerce')

gdp_df = read_data_with_structure(r"D:\GDP_per_Capita(1).xlsx")
gdp_df.rename(columns={'Value': 'GDP_per_Capita'}, inplace=True)
gdp_df['GDP_per_Capita'] = pd.to_numeric(gdp_df['GDP_per_Capita'], errors='coerce')

wage_df = read_data_with_structure(r"D:\Total Average Wage(1).xlsx")
wage_df.rename(columns={'Value': 'Total_Avg_Wage'}, inplace=True)
wage_df['Total_Avg_Wage'] = pd.to_numeric(wage_df['Total_Avg_Wage'], errors='coerce')

fertility_df = read_data_with_structure(r"D:\Fertility_Rate(1).xlsx")
fertility_df.rename(columns={'Value': 'Fertility_Rate'}, inplace=True)
fertility_df['Fertility_Rate'] = pd.to_numeric(fertility_df['Fertility_Rate'], errors='coerce')

fem_emp_df = read_data_with_structure(r"D:\FEM_UNEMP(1).xlsx")
fem_emp_df.rename(columns={'Value': 'FEM_UNEMP'}, inplace=True)
fem_emp_df['FEM_UNEMP'] = pd.to_numeric(fem_emp_df['FEM_UNEMP'], errors='coerce')

hdi_df = read_data_with_structure(r"D:\HDI(1).xlsx")
hdi_df.rename(columns={'Value': 'HDI'}, inplace=True)
hdi_df['HDI'] = pd.to_numeric(hdi_df['HDI'], errors='coerce')

hist_trade_df = read_data_with_structure(r"D:\HIST_TRADE(3).xlsx")
hist_trade_df.rename(columns={'Value': 'HIST_TRADE'}, inplace=True)
hist_trade_df['HIST_TRADE'] = pd.to_numeric(hist_trade_df['HIST_TRADE'], errors='coerce')

gwg_df = read_data_with_structure(r"D:\GWG.xlsx")
gwg_df.rename(columns={'Value': 'GWG'}, inplace=True)
gwg_df['GWG'] = pd.to_numeric(gwg_df['GWG'], errors='coerce')

# 2. 合并数据
data = fdi_df.merge(gdp_df, on=['Country', 'Year'], how='inner')
data = data.merge(wage_df, on=['Country', 'Year'], how='inner')
data = data.merge(fertility_df, on=['Country', 'Year'], how='inner')
data = data.merge(fem_emp_df, on=['Country', 'Year'], how='inner')
data = data.merge(hdi_df, on=['Country', 'Year'], how='inner')
data = data.merge(hist_trade_df, on=['Country', 'Year'], how='inner')
data = data.merge(gwg_df, on=['Country', 'Year'], how='inner')

# 检查数据是否有缺失值
print("数据缺失值检查:")
print(data.isnull().sum())

# 检查数据类型
print("\n数据类型检查:")
print(data.dtypes)

# 检查是否有非数值数据
for col in data.columns:
    if col not in ['Country', 'Year'] and data[col].dtype == 'object':
        print(f"\n警告: {col} 列包含非数值数据")
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 处理缺失值
data = data.dropna()

# 确保Year列是数值型
data['Year'] = pd.to_numeric(data['Year'])

# 3. 设置面板数据结构
data.set_index(['Country', 'Year'], inplace=True)

# 4. 定义回归变量
X_orig = data[['FDI_inflow', 'GDP_per_Capita', 'Total_Avg_Wage', 'HDI', 'Fertility_Rate', 'FEM_UNEMP', 'HIST_TRADE']]

# 标准化数据以减少多重共线性问题
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X_orig)
X_scaled = pd.DataFrame(X_scaled_array, columns=X_orig.columns, index=X_orig.index)

# 检查多重共线性
X_with_const = sm.add_constant(X_scaled)
vif_data = pd.DataFrame()
vif_data["变量"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print("\n多重共线性检验 (VIF) - 标准化后:")
print(vif_data)

# 处理高VIF值变量
high_vif = vif_data[vif_data["VIF"] > 10]
if not high_vif.empty:
    print("\n警告：以下变量存在严重多重共线性问题:")
    print(high_vif)

    # 移除VIF值最高的变量（除了常数项）
    worst_var = high_vif[high_vif["变量"] != "const"].sort_values("VIF", ascending=False)
    if not worst_var.empty:
        var_to_remove = worst_var.iloc[0]["变量"]
        print(f"\n自动移除多重共线性最严重的变量: {var_to_remove}")
        X_scaled = X_scaled.drop(columns=[var_to_remove])
        print(f"剩余变量: {X_scaled.columns.tolist()}")

# 添加常数项
X_scaled = sm.add_constant(X_scaled)
y = data['GWG']

# 5. 首先运行无固定效应模型作为基准
print("\n无固定效应模型:")
try:
    pooled_model = PanelOLS(y, X_scaled, entity_effects=False, time_effects=False)
    pooled_results = pooled_model.fit(cov_type='robust')
    print(pooled_results.summary)
except Exception as e:
    print(f"无固定效应模型估计失败: {str(e)}")
    print("尝试使用check_rank=False...")
    pooled_model = PanelOLS(y, X_scaled, entity_effects=False, time_effects=False)
    pooled_results = pooled_model.fit(cov_type='robust', check_rank=False)
    print(pooled_results.summary)

# 6. 运行仅包含国家固定效应的模型
print("\n仅国家固定效应模型:")
try:
    entity_model = PanelOLS(y, X_scaled, entity_effects=True, time_effects=False)
    entity_results = entity_model.fit(cov_type='robust')
    print(entity_results.summary)
except Exception as e:
    print(f"国家固定效应模型估计失败: {str(e)}")
    print("尝试使用check_rank=False...")
    entity_model = PanelOLS(y, X_scaled, entity_effects=True, time_effects=False)
    entity_results = entity_model.fit(cov_type='robust', check_rank=False)
    print(entity_results.summary)

# 7. 运行仅包含时间固定效应的模型
print("\n仅时间固定效应模型:")
try:
    time_model = PanelOLS(y, X_scaled, entity_effects=False, time_effects=True)
    time_results = time_model.fit(cov_type='robust')
    print(time_results.summary)
except Exception as e:
    print(f"时间固定效应模型估计失败: {str(e)}")
    print("尝试使用check_rank=False...")
    time_model = PanelOLS(y, X_scaled, entity_effects=False, time_effects=True)
    time_results = time_model.fit(cov_type='robust', check_rank=False)
    print(time_results.summary)

# 8. 运行双固定效应模型
print("\n双固定效应模型:")
try:
    both_model = PanelOLS(y, X_scaled, entity_effects=True, time_effects=True)
    both_results = both_model.fit(cov_type='robust')
    print(both_results.summary)
except Exception as e:
    print(f"双固定效应模型估计失败: {str(e)}")
    print("尝试使用check_rank=False...")
    both_model = PanelOLS(y, X_scaled, entity_effects=True, time_effects=True)
    both_results = both_model.fit(cov_type='robust', check_rank=False)
    print(both_results.summary)

# 9. 检验国家固定效应是否应该引入 (F检验)
print("\n检验国家固定效应是否应该引入:")
f_stat_entity = None
p_val_entity = None
try:
    # 计算F统计量
    ssr_restricted = (pooled_results.resids ** 2).sum()
    ssr_unrestricted = (entity_results.resids ** 2).sum()
    df_restricted = pooled_results.df_resid
    df_unrestricted = entity_results.df_resid
    df_diff = df_restricted - df_unrestricted

    if df_diff > 0:
        f_stat_entity = ((ssr_restricted - ssr_unrestricted) / df_diff) / (ssr_unrestricted / df_unrestricted)
        p_val_entity = 1 - stats.f.cdf(f_stat_entity, df_diff, df_unrestricted)
        print(f"F统计量: {f_stat_entity:.4f}, P值: {p_val_entity:.4f}")

        if p_val_entity < 0.05:
            print("结论: 国家固定效应显著，应该引入国家固定效应")
            entity_significant = True
        else:
            print("结论: 国家固定效应不显著，可以不引入国家固定效应")
            entity_significant = False
    else:
        print("无法计算F统计量：自由度差异不足")
        entity_significant = False
except Exception as e:
    print(f"F检验计算失败: {str(e)}")
    entity_significant = False

# 10. 检验时间固定效应是否应该引入 (F检验)
print("\n检验时间固定效应是否应该引入:")
f_stat_time = None
p_val_time = None
try:
    # 计算F统计量
    ssr_restricted = (pooled_results.resids ** 2).sum()
    ssr_unrestricted = (time_results.resids ** 2).sum()
    df_restricted = pooled_results.df_resid
    df_unrestricted = time_results.df_resid
    df_diff = df_restricted - df_unrestricted

    if df_diff > 0:
        f_stat_time = ((ssr_restricted - ssr_unrestricted) / df_diff) / (ssr_unrestricted / df_unrestricted)
        p_val_time = 1 - stats.f.cdf(f_stat_time, df_diff, df_unrestricted)
        print(f"F统计量: {f_stat_time:.4f}, P值: {p_val_time:.4f}")

        if p_val_time < 0.05:
            print("结论: 时间固定效应显著，应该引入时间固定效应")
            time_significant = True
        else:
            print("结论: 时间固定效应不显著，可以不引入时间固定效应")
            time_significant = False
    else:
        print("无法计算F统计量：自由度差异不足")
        time_significant = False
except Exception as e:
    print(f"F检验计算失败: {str(e)}")
    time_significant = False

# 11. 基于F检验结果选择最终模型
print("\n基于F检验结果选择最终模型:")
try:
    if entity_significant and time_significant:
        print("国家和时间固定效应均显著，选择双固定效应模型")
        final_model = both_results
        model_type = "双固定效应模型"
    elif entity_significant:
        print("仅国家固定效应显著，选择国家固定效应模型")
        final_model = entity_results
        model_type = "国家固定效应模型"
    elif time_significant:
        print("仅时间固定效应显著，选择时间固定效应模型")
        final_model = time_results
        model_type = "时间固定效应模型"
    else:
        print("国家和时间固定效应均不显著，选择无固定效应模型")
        final_model = pooled_results
        model_type = "无固定效应模型"
except Exception as e:
    print(f"模型选择失败: {str(e)}")
    print("默认选择双固定效应模型作为最终模型")
    final_model = both_results
    model_type = "双固定效应模型(默认)"

# 12. 输出最终模型结果
print(f"\n最终选择的模型: {model_type}")
print(final_model.summary)

# 13. 保存结果到Excel
results_summary = pd.DataFrame({
    '无固定效应': pooled_results.params,
    '无固定效应_P值': pooled_results.pvalues,
    '国家固定效应': entity_results.params,
    '国家固定效应_P值': entity_results.pvalues,
    '时间固定效应': time_results.params,
    '时间固定效应_P值': time_results.pvalues,
    '双固定效应': both_results.params,
    '双固定效应_P值': both_results.pvalues,
    '最终模型': final_model.params,
    '最终模型_P值': final_model.pvalues
})

# 保存F检验结果
f_test_data = {
    '检验类型': ['国家固定效应F检验', '时间固定效应F检验'],
    'F统计量': [f_stat_entity, f_stat_time],
    'P值': [p_val_entity, p_val_time],
    '是否显著': [entity_significant, time_significant]
}
f_test_results = pd.DataFrame(f_test_data)

# 保存结果
results_summary.to_excel(r'D:\PythonProject\203\203正式\固定效应回归结果.xlsx')
f_test_results.to_excel(r'D:\PythonProject\203\203正式\固定效应F检验结果.xlsx')
print("\n回归结果已保存到Excel文件")

# 14. 输出最终建议
print("\n最终建议:")
print(f"1. 根据F检验结果，应选择{model_type}进行分析。")

if entity_significant:
    print("2. 国家固定效应显著，表明不同国家间存在系统性差异，这些差异会影响因变量。")
else:
    print("2. 国家固定效应不显著，表明不同国家间的系统性差异对因变量影响不大。")

if time_significant:
    print("3. 时间固定效应显著，表明不同时间点存在系统性差异，这些差异会影响因变量。")
else:
    print("3. 时间固定效应不显著，表明不同时间点的系统性差异对因变量影响不大。")

print("4. 在解释结果时，应关注最终模型中显著的变量及其系数方向和大小。")