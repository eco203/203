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
})

# 将 'Country' 和 'Year' 设置为索引，模拟面板数据结构
data.set_index(['Country', 'Year'], inplace=True)

# 查看前几行数据
print(data.head())
