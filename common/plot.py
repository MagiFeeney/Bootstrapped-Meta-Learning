import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from datetime import datetime
from .utils import load_results

mpl.use('Agg')
plt.style.use('bmh')



def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window
    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


def ts2xy(data_frame):
    index = data_frame.columns.values[0]
    y = np.array(data_frame[index].values)
    return y


def count_files_with_prefix(folder_path, prefix):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix):
            count += 1
    return count


def plot(max_steps,
         results_dir,
         smooth_radius,
         log_interval,
         steps_per_rollout,
         algo_name,
         figsize=(10, 5)):

    flag = False if algo_name == "A2C" else True # because A2C doesn't have meta learning
    
    store_dir = "images" # the directory stores images
    os.makedirs(store_dir, exist_ok=True)

    if flag:
        ys = [[], [], []]
    else:
        ys = [[], []]

    num_seeds = count_files_with_prefix(os.path.join(results_dir, algo_name), "seed-")

    for i in range(num_seeds):
        rd = results_dir + f'/{algo_name}/seed-' + str(i)
        data_frames = load_results(rd)
        for j in range(len(ys)):
            y = ts2xy(data_frames[j])
            ys[j].append(y)

    ys = [np.vstack(y) for y in ys]
    
    x = []
    s = []
    s_err = []

    interval = [log_interval] * 3
    interval[-1] = steps_per_rollout
    
    for j in range(len(ys)):
        y = np.mean(ys[j], axis=0)
        y = smooth(y, radius=smooth_radius)
        s.append(y)
        s_err.append(np.std(ys[j], axis=0) / np.sqrt(len(ys[j])))
        x.append(list(range(1, len(y) * interval[j] + 1, interval[j])))
    
    shade_flag = True if num_seeds > 1 else False

    plt_info = {"xlabel": ['Steps', 'Env Steps into Cycle', 'Env Steps into Cycle'],
                "ylabel": ['Cumulative Reward', 'Rew./Step', 'Entropy Rate'],
                "color": ["steelblue", "blue", "mediumorchid"],
                "fill_color": ["lightskyblue", "cornflowerblue", "violet"],
                "file_name": ['cumulative-rewards', 'rew-step', 'entropy-rate'],
                "xlim": [None, [0, 4e5 if max_steps >= 4e5 else max_steps], [0, 4e5 if max_steps >= 4e5 else max_steps]],
                "ylim": [None, [-0.5, 0.5], None]}
    
    for i in range(len(s)):
        fig = plt.figure(figsize=figsize)
        plt.plot(x[i], s[i], color = plt_info["color"][i])
        if shade_flag:
            plt.fill_between(x[i], s[i] - s_err[i], s[i] + s_err[i], alpha=0.2, color=plt_info["fill_color"][i])
            prefix = "shaded"
        else:
            prefix = ""                
        plt.xlim(plt_info["xlim"][i])
        plt.ylim(plt_info["ylim"][i])
        plt.xlabel(plt_info["xlabel"][i])
        plt.ylabel(plt_info["ylabel"][i])
        fig.tight_layout()
        plt.savefig(store_dir + prefix + plt_info["file_name"][i])
        plt.close(fig)        
