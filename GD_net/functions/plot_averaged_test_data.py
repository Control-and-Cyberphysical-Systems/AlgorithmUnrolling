import numpy as np
import os
import matplotlib.pyplot as plt
from functions.helperfunctions import *
from functions.save_as_tikz import save_plot_tikz
def plot_averaged_dict(to_plot:dict ={},load_name:str='test_3.json'):
    '''
    Plot a test series over different number of optimization steps -> needs a number of evaluate_dpc runs beforehand
    plots the values in to_plot otherwise loads them from the file behind 'load_name'
    '''
    if not to_plot['data']:
        to_plot = read_data(load_name)
    data = to_plot['data']
    algo_steps = []
    reformulated_dict = {}
    for key,item in data.items():
        algo_steps.append(int(key))
    first_series = str(algo_steps[0])
    unlearned_keys = data[first_series]['unlearned']
    translator = data[first_series]['key translator']
                
    data_keys = list(data[first_series]['key translator'].keys())
    learn_functions = []
    dist_calcs = []
    for m,translation in data[first_series]['key translator'].items():
        translated = translation['loss fcn']
        translator[f'{m}']['combined'] = f'{m},{translated}'
        learn_functions.append(f'{m},{translated}')
        reformulated_dict[translated] = {}
        for d in data_keys:
            reformulated_dict[translated][translator[d]['model']] = np.zeros(len(algo_steps))
        for d in unlearned_keys:
            reformulated_dict[translated][d] = np.zeros(len(algo_steps))
    inv_fcn = {}
    inv_model = {}
    inv_combinator = {}
    for m,t in translator.items():
        inv_fcn[t['loss fcn']] = m
        inv_model[t['model']] = m
        inv_combinator[t['loss fcn']] = t['combined']
    for k in data[first_series]['0'].keys():
        if k not in learn_functions:
            dist_calcs.append(k)
            reformulated_dict[k] = {}
            for d in data_keys:
                reformulated_dict[k][translator[d]['model']] = np.zeros(len(algo_steps))
            for d in unlearned_keys:
                reformulated_dict[k][d] = np.zeros(len(algo_steps))
    for key,item in reformulated_dict.items():
        compare_fcn = key
        for key2,item2 in item.items():
            model_number = key2
            for ind,algo_number in enumerate(algo_steps):
                if compare_fcn not in dist_calcs:
                    perfect_solution = data[str(algo_number)]['perfect'][inv_fcn[compare_fcn]]
                    if model_number not in unlearned_keys:
                        model_solution = data[str(algo_number)][inv_model[model_number]][inv_combinator[compare_fcn]]
                    else:
                        model_solution = data[str(algo_number)][model_number][inv_combinator[compare_fcn]]
                    item2[ind] = np.log10(np.abs(((perfect_solution - model_solution)/perfect_solution)))
                else:
                    if model_number not in unlearned_keys:
                        distance = data[str(algo_number)][inv_model[model_number]][compare_fcn]
                    else:
                        distance = data[str(algo_number)][model_number][compare_fcn]
                    item2[ind] = np.log10(distance)
    colors = ['firebrick','navy','fuchsia','forestgreen','teal','darkgoldenrod','darkorange','black','blue','red','green']
    for function_key,model_results in reformulated_dict.items():
        fig = plt.figure()
        plt.title(function_key)
        plt.xlabel('algo steps')
        if function_key in inv_fcn.keys():
            ylabel = '$log(\\frac{{f-f^{{\\ast}}}}{{f^{{\\ast}}}})$'
        else:
            ylabel = 'log dist to $(\\cdot)^{{\\ast}}$'
        plt.ylabel(ylabel)
        color_index = 0
        for model_name, model_result in model_results.items():
            if model_name in unlearned_keys:
                plt.plot(algo_steps,model_result,c=colors[color_index],linestyle = '--',label=model_name)
            else:
                plt.plot(algo_steps,model_result,c=colors[color_index],label=model_name)
            plt.scatter(algo_steps,model_result,c=colors[color_index])
            color_index+=1
        plt.legend(loc='lower left',bbox_to_anchor=(1.02, 0))
        own_dic = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        fig.savefig(os.path.join(own_dic,'evaluation_results',f'average_{function_key}.png'), bbox_inches='tight')
        plt.close(fig)
        save_plot_tikz(model_results=model_results,
                       algo_steps=algo_steps,
                       unlearned_keys=unlearned_keys,
                       title=function_key,
                       xlabel='algo steps',
                       ylabel=ylabel,
                       filename=f'average_{function_key}.txt')
    return