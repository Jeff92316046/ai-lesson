import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

weight = ctrl.Antecedent(np.arange(0, 500, 1), 'weight')
age = ctrl.Antecedent(np.arange(0, 150, 1), 'age')
diet = ctrl.Consequent(np.arange(0, 10, 1), 'diet')

weight['輕'] = fuzz.trapmf(weight.universe, [0, 0, 30, 70])
weight['中'] = fuzz.trimf(weight.universe, [40, 60, 90])
weight['重'] = fuzz.trapmf(weight.universe, [60, 110, 120, 120])

age['青'] = fuzz.trapmf(age.universe, [0, 0, 30, 70])
age['中'] = fuzz.trimf(age.universe, [30, 50, 70])
age['老'] = fuzz.trapmf(age.universe, [40, 60, 120, 120])

diet['少'] = fuzz.trapmf(diet.universe, [0, 0, 1, 3])
diet['中'] = fuzz.trimf(diet.universe, [2, 3, 4])
diet['多'] = fuzz.trapmf(diet.universe, [2, 5, 10, 10])

rule1 = ctrl.Rule(weight['輕'] & age['青'], diet['多'])
rule2 = ctrl.Rule(weight['輕'] & age['中'], diet['中'])
rule3 = ctrl.Rule(weight['輕'] & age['老'], diet['少'])
rule4 = ctrl.Rule(weight['中'] & age['老'], diet['少'])
rule5 = ctrl.Rule(weight['中'] & age['中'], diet['中'])
rule6 = ctrl.Rule(weight['重'] & age['老'], diet['少'])
rule7 = ctrl.Rule(weight['重'] & age['中'], diet['中'])

diet_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
diet_sim = ctrl.ControlSystemSimulation(diet_ctrl)

diet_sim.input['weight'] = 78  
diet_sim.input['age'] = 50    

diet_sim.compute()
diet.view(sim=diet_sim)

print(f"飲食量: {diet_sim.output['diet']}")

# weight.view()
# age.view()
# diet.view()

input()