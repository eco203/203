import pandas as pd
import statsmodels.formula.api as smf

def read_and_melt(file_path, var_name):
    """
    读取Excel文件：
      - 假设第一行为各国家名称，第一列为年份（作为索引）。
      - 重置索引后，将宽格式转换成长格式，生成包含 'Year', 'Country', var_name 三列的数据。
    """
    # 读取文件时：header=0, index_col=0使第一行为列名、第一列为索引
    df = pd.read_excel(file_path, header=0, index_col=0)
    df.index.name = 'Year'
    # 重置索引，将 Year 变成普通列
    df = df.reset_index()
    # 去除各列的空格
    df.columns = df.columns.str.strip()
    df['Year'] = df['Year'].astype(str).str.strip()
    # 宽转长：id_vars 为 Year，其余列作为国家数据
    df_long = df.melt(id_vars='Year', var_name='Country', value_name=var_name)
    # 清洗 Country 字段
    df_long['Country'] = df_long['Country'].astype(str).str.strip()
    # 将 Year 转换为数值型
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    return df_long

# 1. 读取各文件（请根据实际文件路径修改）
df_gwg  = read_and_melt(r"D:\GWG.xlsx", 'GWG')
df_fdi  = read_and_melt(r"D:\FDI_Percentage_of_GDP__2000_2022__Fixed_.xlsx", 'FDI')
df_gdp  = read_and_melt(r"D:\GDP.PCAP.CD.xlsx", 'GDP_PC')
df_inc  = read_and_melt(r"D:\INC.xlsx", 'TAW')      # 假定 INC 表示 Total Average Wage
df_hdi  = read_and_melt(r"D:\HDI_Data.xlsx", 'HDI')
df_fert = read_and_melt(r"D:\Fertility.xlsx", 'FERT')
df_fem  = read_and_melt(r"D:\FEM_EMP.xlsx", 'FEM_EMP')  # 中介变量：女性就业份额

# 2. 合并所有数据（以 Year 与 Country 为键，使用 outer 合并方便查缺）
df_panel = df_gwg.merge(df_fdi, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_gdp, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_inc, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_hdi, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_fert, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_fem, on=['Year', 'Country'], how='outer')
df_panel = df_panel.sort_values(['Year', 'Country'])

print("合并后的面板数据预览:")
print(df_panel.head(10))
print("各变量缺失统计:")
print(df_panel[['GWG','FDI','GDP_PC','TAW','HDI','FERT','FEM_EMP']].isnull().sum())

# 3. 缺失值处理：按 Country 分组对每个数值变量进行线性插值，再前向填充与后向填充
numeric_vars = ['GWG', 'FDI', 'GDP_PC', 'TAW', 'HDI', 'FERT', 'FEM_EMP']
df_panel_filled = df_panel.copy()

for var in numeric_vars:
    # 使用 transform 确保输出索引与原数据对齐
    df_panel_filled[var] = df_panel_filled.groupby('Country')[var]\
        .transform(lambda group: group.interpolate(method='linear'))
    df_panel_filled[var] = df_panel_filled.groupby('Country')[var]\
        .transform(lambda group: group.ffill().bfill())

# 删除关键变量仍存在缺失值的行
df_panel_filled = df_panel_filled.dropna(subset=numeric_vars)

print("缺失值处理后的数据预览:")
print(df_panel_filled.head(10))
print("处理后缺失统计:")
print(df_panel_filled[numeric_vars].isnull().sum())
print("Unique Countries:", df_panel_filled['Country'].unique())
print("Unique Years:", df_panel_filled['Year'].unique())

# 4. 机制检验

# (1) 第一阶段：检验 FDI 对中介变量 FEM_EMP 的影响
formula_med = 'FEM_EMP ~ FDI + GDP_PC + TAW + HDI + FERT + C(Country) + C(Year)'
model_med = smf.ols(formula=formula_med, data=df_panel_filled).fit()
print("\n【机制检验 - 第一阶段】FEM_EMP 回归结果 (FDI → FEM_EMP):")
print(model_med.summary())

# (2) 第二阶段：检验中介变量 FEM_EMP 对 GWG 的影响（同时控制 FDI 及其它变量）
formula_out = 'GWG ~ FEM_EMP + FDI + GDP_PC + TAW + HDI + FERT + C(Country) + C(Year)'
model_out = smf.ols(formula=formula_out, data=df_panel_filled).fit()
print("\n【机制检验 - 第二阶段】GWG 回归结果 (FEM_EMP → GWG):")
print(model_out.summary())
