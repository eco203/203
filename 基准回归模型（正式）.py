import pandas as pd
import statsmodels.formula.api as smf

def read_and_melt(file_path, var_name):
    """
    读取Excel文件，假设第一行为国家名称，第一列为年份（作为索引），
    然后将宽格式转换成长格式（包含Year, Country, var_name）
    """
    # 使用 header=0 和 index_col=0 读取文件
    df = pd.read_excel(file_path, header=0, index_col=0)
    df.index.name = 'Year'
    # 重置索引，将 Year 变为普通列
    df = df.reset_index()
    # 去除列名和数据中的空格
    df.columns = df.columns.str.strip()
    df['Year'] = df['Year'].astype(str).str.strip()
    # 转换成长格式
    df_long = df.melt(id_vars='Year', var_name='Country', value_name=var_name)
    df_long['Country'] = df_long['Country'].astype(str).str.strip()
    # 将 Year 转换为数值型
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    return df_long

# 读取各个文件（请根据实际路径修改）
df_gwg  = read_and_melt(r"D:\GWG.xlsx", 'GWG')
df_fdi  = read_and_melt(r"D:\FDI_Percentage_of_GDP__2000_2022__Fixed_.xlsx", 'FDI')
df_gdp  = read_and_melt(r"D:\GDP.PCAP.CD.xlsx", 'GDP_PC')
df_inc  = read_and_melt(r"D:\INC.xlsx", 'TAW')   # 假设 INC 表示 Total Average Wage
df_hdi  = read_and_melt(r"D:\HDI_Data.xlsx", 'HDI')
df_fert = read_and_melt(r"D:\Fertility.xlsx", 'FERT')

# 依次合并各数据集（以 Year 和 Country 为键）
df_panel = df_gwg.merge(df_fdi, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_gdp, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_inc, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_hdi, on=['Year', 'Country'], how='outer')
df_panel = df_panel.merge(df_fert, on=['Year', 'Country'], how='outer')
df_panel = df_panel.sort_values(['Year', 'Country'])

print("合并后的面板数据预览:")
print(df_panel.head(10))

# 检查缺失情况
print("缺失统计:")
print(df_panel[['GWG', 'FDI', 'GDP_PC', 'TAW', 'HDI', 'FERT']].isnull().sum())

# 缺失值处理：按 Country 分组对每个数值变量进行线性插值，然后前向填充和后向填充
numeric_vars = ['GWG', 'FDI', 'GDP_PC', 'TAW', 'HDI', 'FERT']
df_panel_filled = df_panel.copy()

for var in numeric_vars:
    # 使用 transform 保证输出与原始 DataFrame 索引对齐
    df_panel_filled[var] = df_panel_filled.groupby('Country')[var].transform(lambda group: group.interpolate(method='linear'))
    df_panel_filled[var] = df_panel_filled.groupby('Country')[var].transform(lambda group: group.ffill().bfill())


print("缺失值处理后的数据预览:")
print(df_panel_filled.head(10))
print("处理后缺失统计:")
print(df_panel_filled[numeric_vars].isnull().sum())

# 检查 Country 与 Year 的唯一值
print("Unique Countries:", df_panel_filled['Country'].unique())
print("Unique Years:", df_panel_filled['Year'].unique())

# 构建基准回归模型：GWG ~ FDI + GDP_PC + TAW + HDI + FERT + 固定效应（Country 和 Year）
formula = 'GWG ~ FDI + GDP_PC + TAW + HDI + FERT + C(Country) + C(Year)'
model = smf.ols(formula=formula, data=df_panel_filled).fit()

print("\n基准回归模型结果摘要:")
print(model.summary())






