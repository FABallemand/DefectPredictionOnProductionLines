import pandas as pd
import utils

train_input, train_output = utils.loadTrainingData()

# train_input_balanced, train_output_balanced = utils.balanceClassesByDuplicating(train_input, train_output)
# train_input_balanced.to_csv("data/train_inputs_balanced.csv", index=False, encoding="utf-8")
# train_output_balanced.to_csv("data/train_output_balanced.csv", index=False, encoding="utf-8")

train_input_balanced_mod, train_output_balanced_mod = utils.balanceClassesByDuplicating(train_input, train_output, modify=True, id=True, nb_features=13, max_modif_rate=0.005)
train_input_balanced_mod.to_csv("data/train_inputs_balanced_mod_005.csv", index=False, encoding="utf-8")
train_output_balanced_mod.to_csv("data/train_output_balanced_mod_005.csv", index=False, encoding="utf-8")