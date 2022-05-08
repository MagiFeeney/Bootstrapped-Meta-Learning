import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from datetime import datetime

mpl.use('Agg')
plt.style.use('bmh')

def plot(max_steps, s1, s2, s1_std=None, s2_std=None, shade_flag=False):
 
    current_time = datetime.now()
    current_time = current_time.strftime("%d-%m-%Y %H:%M:%S")
    script_dir = os.path.dirname(__file__)
    child_dir  = "results/"
    folder_name = "results_a2c(" + current_time + ")/" 
    results_dir = os.path.join(script_dir, child_dir, folder_name)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    
    fig1 = plt.figure(figsize=(10, 5)) 
    plt.plot(s1, color = "steelblue")
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    fig1.tight_layout()
    plt.savefig(results_dir + 'cumulative-rewards')
    plt.close(fig1)
    
    fig3 = plt.figure(figsize=(10, 5)) 
    plt.plot(s2, color = "black")
    plt.xlim([0, 4e5 if max_steps >= 4e5 else max_steps])
    plt.ylim([-0.5, 0.5])
    plt.xlabel('Env Steps into Cycle')
    plt.ylabel('Rew./Step')
    fig3.tight_layout()
    plt.savefig(results_dir + 'rew-step')
    plt.close(fig3)

    if shade_flag:
        x1 = list(range(1, len(s1) + 1))
        x2 = list(range(1, len(s2) + 1))
        fig2 = plt.figure(figsize=(10, 5)) 
        plt.plot(s1, color = "steelblue")
        plt.plot(s1 - s1_std, color = "cornflowerblue", linewidth=0.2)
        plt.plot(s1 + s1_std, color = "cornflowerblue", linewidth=0.2)
        plt.fill_between(x1, s1 - s1_std, s1 + s1_std, alpha=0.2, color = "lightskyblue")
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        fig2.tight_layout()
        plt.savefig(results_dir + 'shaded-cumulative-rewards')
        plt.close(fig2)


        fig4 = plt.figure(figsize=(10, 5)) 
        plt.plot(s2, color = "black")
        plt.fill_between(x2, s2 - s2_std, s2 + s2_std, alpha=0.2, color = "grey")
        plt.xlim([0, 4e5 if max_steps >= 4e5 else max_steps])
        plt.ylim([-0.5, 0.5])
        plt.xlabel('Env Steps into Cycle')
        plt.ylabel('Rew./Step')
        fig4.tight_layout()
        plt.savefig(results_dir + 'shaded-rew-step-1')
        plt.close(fig4)

        fig6 = plt.figure(figsize=(10, 5)) 
        plt.plot(s2, color = "black")
        plt.fill_between(x2, s2 - s2_std, s2 + s2_std, alpha=0.2, color = "grey")
        plt.xlim([0, 4e5 if max_steps >= 4e5 else max_steps])
        plt.ylim([0, 0.27])
        plt.yticks([0.05, 0.15, 0.25])
        plt.xlabel('Env Steps into Cycle')
        plt.ylabel('Rew./Step')
        fig6.tight_layout()
        plt.savefig(results_dir + 'shaded-rew-step-2')
        plt.close(fig6)