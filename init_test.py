"""
learning rate: 0.001, 0.005, 0.025, 0.125
iterations: 1, 3, 9
model depth: 1, 5, 10
model width: 25, 50, 100
extra model: 200x5, 1000x1, 10000x1
N = 20
"""

import json

learning_rates = [0.01, 0.005, 0.025, 0.125]
iterations = [1, 3, 9]
model_depth = [1, 5, 10]
model_width = [25, 50, 100]

models = ["dense_{0}x{1}".format(w,d) for w in model_width for d in model_depth]
models.append("dense_200x5")
models.append("dense_1000x1")
models.append("dense_10000x1")

test_plan = ["l{0}i{1}m{2}".format(l, i, m) for l in learning_rates for i in iterations for m in models]

test_dict = dict()

for test in test_plan:
    test_dict[test] = 0

file = open("test_plan.json", "w")
json.dump(test_dict, file, indent="")
file.close()