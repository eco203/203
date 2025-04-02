import pandas as pd
from linearmodels.iv import IV2SLS

def read_and_melt(file_path, var_name):
    """
    读取 Excel 文件（假设第一行为各国家名称，第一列为年份），
    将数据转换成长格式，返回包含 'Year', 'Country', var_name 的 DataFrame
    """
    # 读取时使用 header=0, index_col=0，使第一行为列名，第一列为索引（年份）
    df = pd.read_excel(file_path, header=0, index_col=0)
    df.index.name = 'Year'
    df = df.reset_index()  # 将 Year 从索引转为普通列
    # 去除各列名称和数据前后的空格
    df.columns = df.columns.str.strip()
    df['Year'] = df['Year'].astype(str).str.strip()
    # 转换成长格式：id_vars 为 Year，其他列为各国家数据
    df_long = df.melt(id_vars='Year', var_name='Country', value_name=var_name)
    df_long['Country'] = df_long['Country'].astype(str).str.strip()
    # 将 Year 转换为数值型
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    return df_long

# 读取各个文件（请根据实际路径修改）
df_gwg         = read_and_melt(r"D:\GWG.xlsx", 'GWG')
df_fdi         = read_and_melt(r"D:\FDI_Percentage_of_GDP__2000_2022__Fixed_.xlsx", 'FDI')
df_gdp         = read_and_melt(r"D:\GDP.PCAP.CD.xlsx", 'GDP_PC')
df_inc         = read_and_melt(r"D:\INC.xlsx", 'TAW')         # 假定 INC 为 Total Average Wage
df_hdi         = read_and_melt(r"D:\HDI_Data.xlsx", 'HDI')
df_fert        = read_and_melt(r"D:\Fertility.xlsx", 'FERT')
df_hist_trade  = read_and_melt(r"D:\HIST_TRADE.xlsx", 'HIST_TRADE')

# 合并各个 DataFrame，按 Year 与 Country 键外连接
df_panel = df_gwg.merge(df_fdi, on=["Year", "Country"], how="outer")
df_panel = df_panel.merge(df_gdp, on=["Year", "Country"], how="outer")
df_panel = df_panel.merge(df_inc, on=["Year", "Country"], how="outer")
df_panel = df_panel.merge(df_hdi, on=["Year", "Country"], how="outer")
df_panel = df_panel.merge(df_fert, on=["Year", "Country"], how="outer")
df_panel = df_panel.merge(df_hist_trade, on=["Year", "Country"], how="outer")
df_panel = df_panel.sort_values(["Year", "Country"])

print("合并后的面板数据预览:")
print(df_panel.head(10))

# 缺失值处理：对数值型变量按 Country 分组进行线性插值，然后前向填充和后向填充
numeric_vars = ["GWG", "FDI", "GDP_PC", "TAW", "HDI", "FERT", "HIST_TRADE"]
df_panel_filled = df_panel.copy()
for var in numeric_vars:
    # 使用 transform 保证返回值的索引与原数据一致
    df_panel_filled[var] = df_panel_filled.groupby("Country")[var].transform(lambda group: group.interpolate(method="linear"))
    df_panel_filled[var] = df_panel_filled.groupby("Country")[var].transform(lambda group: group.ffill().bfill())
# 删除仍存在缺失的观测
df_panel_filled = df_panel_filled.dropna(subset=numeric_vars)

print("缺失值处理后的数据预览:")
print(df_panel_filled.head(10))
print("处理后缺失统计:")
print(df_panel_filled[numeric_vars].isnull().sum())
print("Unique Countries:", df_panel_filled["Country"].unique())
print("Unique Years:", df_panel_filled["Year"].unique())

# 内生性检验：使用工具变量 HIST_TRADE 对 FDI 进行 2SLS 估计
# 模型公式：GWG ~ GDP_PC + TAW + HDI + FERT + C(Country) + C(Year) + [FDI ~ HIST_TRADE]
iv_formula = "GWG ~ GDP_PC + TAW + HDI + FERT + C(Country) + C(Year) + [FDI ~ HIST_TRADE]"

# 使用 IV2SLS 的公式接口进行 2SLS 回归（设置 robust 标准误）
iv_results = IV2SLS.from_formula(iv_formula, data=df_panel_filled).fit(cov_type='robust')
print("\n【内生性检验】IV回归结果摘要:")
print(iv_results.summary)
