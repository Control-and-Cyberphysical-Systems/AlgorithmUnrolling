# AlgorithmUnrolling

Training and evaluation of ADMM / Gradient-Descent unrolling networks for optimal control problems.

## Repository Structure
In the folder gd_net, all files are contained, that are necessary for the AU-Nets to run.
- \GD_net\models\...: Definition of the learnable au models
- \GD_net\functions: functions from the network
  - ...\evaluate_dpc: runs the evaluation of the trained network and produces the most plots
  - ...\train: runs the training for one network
  - ...\test: tests the trained network against a test-dataset (for a sanity check)
  - ...\save_as_tikz / ...\plot_averaged_test_data: produces plots and tikz-files after evaluation
  - ...\models: chooses, which model to select for the network. NEEDS TO BE UPDATED, IF A NEW MODEL IS ADDED!
  - ...\helperfunctions: multiple smaller and bigger functions with quality of life features as well as the untrained solvers
- \GD_net\loss_functions: contains all the loss functions for training and evaluation
- \GD_net\gd_net: Class that combines all the subfunctiions necessary to construct a network
- \model_parameters: saves the trained parameters of the solvers
- \data: contains the data sets on which to train the models on
- \testing_figures, \tikz, \evaluation_results: containes figures and plots after testing/evaluation  
- requirements.txt: has all important libraries
plaintextCode kopieren
## Installation

Requires **Python ≥ 3.11**.

```bash
# 1. Clone the repository
git clone https://github.com/Control-and-Cyberphysical-Systems/AlgorithmUnrolling.git
cd AlgorithmUnrolling

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .\.venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
GPU (optional): If you have a CUDA-capable GPU, install the matching torch build, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu121

Running the Program
bashCode kopierenpythonrunner.py
By default the script performs both training and evaluation. To run only one part, edit the flags at the top of run_experiment.py:
pythonCode kopierentraining   = True    # set to False to skip training
evaluating = True    # set to False to skip evaluationConfiguration
All parameters are defined at the top of run_experiment.py. The three most important ones:
```
# Run the program
To run the program, the \GD_net\runner.py file needs to be run.
## Parameters - can be adjusted
| Parameter | Description | Example | Typical range|
|---|---|---|---|
| lrs | Learning Rates throughout the training | [1e-3, 1e-4] | 1e-2 ... 1e-5|
| n_p | Number of data points along one axis (there are two) | 20 | 10 ... 200 |
| test_pars | Optimization step numbers to test | [5,10,15] | 10 ... 200 |
| N | Prediction horizon of the problem (needs suitable matlab parameters) | 8 | 2 ... 200 |
| b_s | Batch size of the training 0 -> Full set for one epoch | 0 | 0 ... 0.799*(n_p**2) |
| repetitions | how long to train at what learning rate in lrs | {1e-3:200,1e-4:2000} | 200...10000|
| tikz | boolean to decide, if tikz files should be generated | False | True ... False |
| printing | boolean to decide, if live output should be printed in ther terminal | True | True ... False |
| evaluating | boolean to decide, if the evaluate_dpc should be run | False | True ... False |
| training | boolean to decide, if the networks should be trained | True | True ... False |


# Support
Bugs / Questions: Open an Issue
Contributions: Fork the repository and submit a Pull Request
