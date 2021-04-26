import glob
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools

def save_output(sub_data,evals_result=None,figure=None):
    sub_count = len(glob.glob('../output/*'))
    dir_name = ('../output/sub_{:0=3}'.format(sub_count))
    os.mkdir(dir_name)
    sub_data.to_csv(dir_name+'/sub_{:0=3}.csv'.format(sub_count),index=False)
    if(figure != None):
        figure.savefig(dir_name+'/img.jpg')
    if(evals_result!=None):
        with open(dir_name+'/evals_result.json', 'w') as outfile:
            json.dump(evals_result,outfile,indent=4)
    return