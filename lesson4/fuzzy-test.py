import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
velocity = ctrl.Antecedent(np.arange(0,200,1),'velocity')
velocity.automf(3)
velocity['poor'].view()
a = input()