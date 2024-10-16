import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

X = ctrl.Antecedent(np.arange(0, 2, 1), "X")
Y = ctrl.Antecedent(np.arange(-5, 6, 1), "Y")
Z = ctrl.Consequent(np.arange(-10, 11, 1), "Z")

X["small"] = fuzz.trimf(X.universe, [0, 0, 1])
X["large"] = fuzz.trimf(X.universe, [0, 1, 1])

Y["small"] = fuzz.trimf(Y.universe, [-5, -5, 5])
Y["large"] = fuzz.trimf(Y.universe, [-5, 5, 5])

Z["nl"] = fuzz.trimf(Z.universe, [-10, -10, 10])
Z["ns"] = fuzz.trimf(Z.universe, [-10, -5, 0])
Z["ps"] = fuzz.trimf(Z.universe, [0, 5, 10])
Z["pl"] = fuzz.trimf(Z.universe, [-10, 10, 10])

rule1 = ctrl.Rule(X["small"] & Y["small"], Z["nl"])
rule2 = ctrl.Rule(X["small"] & Y["large"], Z["ns"])
rule3 = ctrl.Rule(X["large"] & Y["small"], Z["ps"])
rule4 = ctrl.Rule(X["large"] & Y["large"], Z["pl"])

Z_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
Z_sim = ctrl.ControlSystemSimulation(Z_ctrl)

Z_sim.input["X"] = 0.5
Z_sim.input["Y"] = -1

Z_sim.compute()
Z.view(sim=Z_sim)
Y.view()
X.view()
print(f"飲食量: {Z_sim.output['Z']}")

# X.view()
# Y.view()
# Z.view()

input()
