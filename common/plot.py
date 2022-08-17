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


def plot(max_steps,
         results_dir,
         num_seeds,
         smooth_radius,
         log_interval,
         steps_per_rollout,
         figsize=(10, 5),
         flag=False):
 
    script_dir = os.path.dirname(os.path.abspath(results_dir))
    store_dir = os.path.join(script_dir, "images/")   # the directory stores images

    if not os.path.isdir(store_dir):
        os.makedirs(store_dir)

    if flag:
        ys = [[], [], []]
    else:
        ys = [[], []]

    for i in range(num_seeds):
        rd = results_dir + '/seed-' + str(i)
        data_frames = load_results(rd)
        for i in range(len(ys)):
            y = ts2xy(data_frames[i])
            ys[i].append(y)

    ys = [np.vstack(y) for y in ys]

    s1 = np.mean(ys[0], axis=0)
    s2 = np.mean(ys[1], axis=0)
    
    s1_err = np.std(ys[0], axis=0) / np.sqrt(len(ys[0]))
    s2_err = np.std(ys[1], axis=0) / np.sqrt(len(ys[1]))
    
    s1 = smooth(s1, radius=smooth_radius)
    s2 = smooth(s2, radius=smooth_radius)

    x1 = list(range(1, len(s1) * log_interval + 1, log_interval))
    x2 = list(range(1, len(s2) * log_interval + 1, log_interval))

    if flag: # entropy rate
        s3 = np.mean(ys[2], axis=0)
        s3_err = np.std(ys[2], axis=0) / np.sqrt(len(ys[2]))
        s3 = smooth(s3, radius=smooth_radius)
        x3 = list(range(1, len(s3) * steps_per_rollout + 1, steps_per_rollout))    

    shade_flag = True if len(ys[0]) > 1 else False

    fig1 = plt.figure(figsize=figsize)
    plt.plot(x1, s1, color = "steelblue")
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    fig1.tight_layout()
    plt.savefig(store_dir + 'cumulative-rewards')
    plt.close(fig1)
    
    fig2 = plt.figure(figsize=figsize) 
    plt.plot(x2, s2, color = "blue")
    plt.xlim([0, 4e5 if max_steps >= 4e5 else max_steps])
    plt.ylim([-0.5, 0.5])
    plt.xlabel('Env Steps into Cycle')
    plt.ylabel('Rew./Step')
    fig2.tight_layout()
    plt.savefig(store_dir + 'rew-step')
    plt.close(fig2)

    if flag:
        fig3 = plt.figure(figsize=figsize) 
        plt.plot(x3, s3, color = "mediumorchid")
        plt.xlim([0, 4e5 if max_steps >= 4e5 else max_steps])
        plt.xlabel('Env Steps into Cycle')
        plt.ylabel('Entropy Rate')
        fig3.tight_layout()
        plt.savefig(store_dir + 'entropy-rate')
        plt.close(fig3)

    if shade_flag:

        fig4 = plt.figure(figsize=figsize) 
        plt.plot(x1, s1, color = "steelblue")
        plt.plot(s1 - s1_err, color = "cornflowerblue", linewidth=0.2)
        plt.plot(s1 + s1_err, color = "cornflowerblue", linewidth=0.2)
        plt.fill_between(x1, s1 - s1_err, s1 + s1_err, alpha=0.2, color = "lightskyblue")
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        fig4.tight_layout()
        plt.savefig(store_dir + 'shaded-cumulative-rewards')
        plt.close(fig4)

        
        fig5 = plt.figure(figsize=figsize) 
        plt.plot(x2, s2, color = "blue")
        plt.fill_between(x2, s2 - s2_err, s2 + s2_err, alpha=0.2, color = "cornflowerblue")
        plt.xlim([0, 4e5 if max_steps >= 4e5 else max_steps])
        plt.ylim([-0.5, 0.5])
        plt.xlabel('Env Steps into Cycle')
        plt.ylabel('Rew./Step')
        fig5.tight_layout()
        plt.savefig(store_dir + 'shaded-rew-step')
        plt.close(fig5)

        if flag:
            fig6 = plt.figure(figsize=figsize) 
            plt.plot(x3, s3, color = "mediumorchid")
            plt.fill_between(x3, s3 - s3_err, s3 + s3_err, alpha=0.2, color = "violet")
            plt.xlim([0, 4e5 if max_steps >= 4e5 else max_steps])
            plt.xlabel('Env Steps into Cycle')
            plt.ylabel('Entropy Rate')
            fig6.tight_layout()
            plt.savefig(store_dir + 'shaded-entropy-rate')
            plt.close(fig6)

        
