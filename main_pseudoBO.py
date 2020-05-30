import json
import numpy as np
import os
import sys
import torch
from time import strftime
from datasets import ActiveSineData
from pseudo_bo import PseudoBO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main_experiment.py <path_to_config>"))
config_path = sys.argv[1]

# Create a folder to store experiment results
timestamp = strftime("%Y-%m-%d_%H-%M")
directory = "results_{}".format(timestamp)
if not os.path.exists(directory):
    os.makedirs(directory)

# Open config file
with open(config_path) as config_file:
    config = json.load(config_file)

# Save config file in experiment directory
with open(directory + '/config.json', 'w') as config_file:
    json.dump(config, config_file)

batch_size = config["batch_size"]
h_dim = config["h_dim"]
x_dim = config["x_dim"]
y_dim = config["y_dim"]
lr = config["lr"]
epochs = config["epochs"]


dataset = ActiveSineData()

pbo = PseudoBO(dataset.inquery, x_dim=1, y_dim=1, h_dim=32, lr=0.1, device=device)

# main loop
for epoch in range(epochs):
    print("Epoch {}{}{}".format('-'*30, epoch, '-'*30))
    pbo.acquisition(3)
