import numpy as np
from gd_net import GD_net
from functions.helperfunctions import save_data
from functions.plot_averaged_test_data import plot_averaged_dict
# Define, which tests to run

## models:
# admm base
# admm dual
# gd dpc dual
# gd dpc prim

# NEW
# dpc mega dual
# dpc mega dual last
# dpc mega
# dpc mega last

N = 8 # -> Better have the matlab files prepared for the number you wanna run
n_p = 70 # -> number of points per axis in the data set. best between 10 and 100 -> for tikz: lower is better
b_s = 0 # -> set the batch size. 0 means full set as batch. If not on a server with gpup or for unstable training, a smaller batch size is recommended -> just set less than int(0.6*(n_p**2))
n_pars = 20
# Set up the dicts for all models you want to test
run_dict_one = {"N":N, # Prädiktionshorizont
                "n_p":n_p, # Punkte im Primal Gatter
                "model":"admm dual", # Model
                "number of model parameters":n_pars,# Predicted gd-steps
                "batch size":b_s, # -> 0 means full training set as a batch
                "repetitions":1000, # Anzahl an wiederholungen der trainingsbatches
                "loss function":"dpc mega dual last", # Name der Kostenfunktion
                "learning rate":1e-2, # Intital learning Rate
                "pre training":True} # should pre-training be considered?

run_dict_two = {"N":N, # Prädiktionshorizont
                "n_p":n_p, # Punkte im Primal Gatter
                "model":"admm base", # Model
                "number of model parameters":n_pars,# Predicted gd-steps
                "batch size":b_s, # -> 0 means full training set as a batch
                "repetitions":1000, # Anzahl an wiederholungen der trainingsbatches
                "loss function":"dpc mega last", # Name der Kostenfunktion
                "learning rate":1e-2, # Intital learning Rate
                "pre training":True} # should pre-training be considered?

run_dict_three = {"N":N, # Prädiktionshorizont
                 "n_p":n_p, # Punkte im Primal Gatter
                 "model":"gd dpc dual", # Model
                 "number of model parameters":n_pars,# Predicted gd-steps
                 "batch size":b_s, # -> 0 means full training set as a batch
                 "repetitions":1000, # Anzahl an wiederholungen der trainingsbatches
                 "loss function":"dpc mega dual last", # Name der Kostenfunktion
                 "learning rate":1e-2, # Intital learning Rate
                 "pre training":True} # should pre-training be considered?

run_dict_four = {"N":N, # Prädiktionshorizont
                 "n_p":n_p, # Punkte im Primal Gatter
                 "model":"gd dpc prim", # Model
                 "number of model parameters":n_pars,# Predicted gd-steps
                 "batch size":b_s, # -> 0 means full training set as a batch
                 "repetitions":1000, # Anzahl an wiederholungen der trainingsbatches
                 "loss function":"dpc mega last", # Name der Kostenfunktion
                 "learning rate":1e-2, # Intital learning Rate
                 "pre training":True} # should pre-training be considered?

# Choose which models to compare
# example: r_dicts = [run_dict_one,run_dict_two,run_dict_three,run_dict_four]
r_dicts = [run_dict_one,run_dict_two,run_dict_three,run_dict_four]
# choose, which learning rates to compare
# example: lrs =     [1e-2,1e-3,1e-4,1e-5]
lrs =     [1e-3,1e-5]
# Vary between the numbers of optimization steps
# example: test_pars = [2,5,10,15,20,25,35,45,55,65,75]
test_pars = list(range(3,25,3))
# How many epochs does the training get
# example: repetitions = {1e-2:200,1e-3:1500,1e-4:7500,1e-5:10000}
repetitions = {1e-2:500,1e-3:1000,1e-5:4000}
# What is the name of the evaluate-test at the end
num = np.prod(np.array(test_pars))
test_name= f"full_test_{len(test_pars)}_{sum(test_pars)}_{sum(int(d) for d in str(num))}.json"
# Set up the data collection
full_data_dict = {"Name":test_name, 
                  "data":{}}
# Do you want to train?
training =  False
# Do you want to evaluate?
evaluating = True
# Do you want your output printed in the console?
printing = True
# Do you want some tikz-graphics? (only recommended for n_p<30)
tikz = False
# How many dual steps do the primal solvers get to iterate a dual solution to their primal one
dual_steps = 1
# track how long to train or evaluate
train_number = 1
eval_number  = 1
# for all test parameters
for tp in test_pars:
    # change the training sets accordingly
    for ri,rd in  enumerate(r_dicts):
        rd["number of model parameters"] = tp
    # If you want to train, start now
    if training:
        for ri,rd in  enumerate(r_dicts):
            for lri in lrs:
                rd["learning rate"] = lri
                if lri in repetitions.keys():
                    rd["repetitions"] = repetitions[lri]
                else:
                    print("No suitable number of epochs given. Go for 200")
                    rd["repetitions"] = 200
                print(100*"--")
                print(f"Training number {train_number} out of {len(r_dicts)*len(test_pars)*len(lrs)} | {rd["model"]} | lr: {lri :.2e} | epchos: {repetitions[lri]:.2e}")
                mo = GD_net(run_dict=rd)
                mo.model.dual_n_steps = dual_steps
                mo.train(mo,printer= printing)
                mo.test(mo, printer = printing)
                mo = None
                train_number+=1
    # if you want to evaluate, start now
    if evaluating:
        print(100*"--")
        print(f"Evaluation number {eval_number} out of {len(test_pars)}")
        eval_number+=1
        otters = []
        n_one = GD_net(run_dict=r_dicts[0])
        n_one.model.dual_n_steps = dual_steps
        for i in range(1,len(r_dicts)):
            n_n = GD_net(run_dict=r_dicts[i])
            n_n.model.dual_n_steps = dual_steps
            otters.append(n_n)
        full_data_dict["data"][str(tp)] = n_one.evaluate_dpc(self=n_one,
                                                        others=otters,
                                                        tikz = tikz,
                                                        printer=printing)
        n_one = None
        otters = None
        # Save the results
        save_data(full_data_dict)
# Plot the averaged values over different number of optimization steps
if evaluating:
    plot_averaged_dict(to_plot=full_data_dict,load_name=test_name)
