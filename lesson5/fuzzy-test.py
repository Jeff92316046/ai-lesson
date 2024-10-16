import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 新建 Antecedent/Consequent 對象，用來保存輸入變數和隸屬度函數
weight = ctrl.Antecedent(np.arange(0, 500, 1), 'weight')
age = ctrl.Antecedent(np.arange(0, 150, 1), 'age')
diet = ctrl.Consequent(np.arange(0, 10, 1), 'diet')

# 自動生成隸屬度函數 (如果不使用自定義隸屬度函數的話)
# quality.automf(3)
# service.automf(3)

# 自定義 Antecedent 的隸屬度函數 (輸入)
weight['輕'] = fuzz.trapmf(weight.universe, [0, 0, 30, 70])
weight['中'] = fuzz.trimf(weight.universe, [40, 60, 90])
weight['重'] = fuzz.trapmf(weight.universe, [60, 110, 120, 120])

age['青'] = fuzz.trapmf(age.universe, [0, 0, 30, 70])
age['中'] = fuzz.trimf(age.universe, [30, 50, 70])
age['老'] = fuzz.trapmf(age.universe, [40, 60, 120, 120])

# 自定義 Consequent 的隸屬度函數 (輸出)
diet['少'] = fuzz.trapmf(diet.universe, [0, 0, 1, 3])
diet['中'] = fuzz.trimf(diet.universe, [2, 3, 4])
diet['多'] = fuzz.trapmf(diet.universe, [2, 5, 10, 10])

# 查看隸屬度函數
weight.view()
age.view()
diet.view()

input()