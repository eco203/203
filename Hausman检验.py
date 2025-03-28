# 随机效应回归模型
from linearmodels.panel import RandomEffects

# 随机效应模型：注意这里不在公式中加入 "RandomEffects"
random_effect_model = RandomEffects.from_formula('GWG ~ FDI + GDP_per_capita + Total_average_wage + HDI + Fertility', data)
random_effect_result = random_effect_model.fit()

# Hausman检验：比较固定效应模型和随机效应模型
from linearmodels.panel import compare
results = compare({'Fixed Effects': fixed_effect_result, 'Random Effects': random_effect_result})
print(results)

