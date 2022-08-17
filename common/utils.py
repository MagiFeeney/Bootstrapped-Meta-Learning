import csv
import json
import os
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pandas
import torch

class storage():
    
    EXT1 = "monitor1.csv"
    EXT2 = "monitor2.csv"
    EXT3 = "monitor3.csv"

    def __init__(
        self,
        max_steps,
        log_interval,
        filename: Optional[str] = None,
    ):
        self.max_steps = max_steps
        self.flag      = True if max_steps > 4e5 else False
        self.step      = 1
        assert log_interval <= 4e5, "Log interval too large to work."
        self.log_interval = log_interval
        
        if filename is not None:
            self.results_writer = ResultsWriter(
                filename
            )
        else:
            self.results_writer = None

    def insert(self, cumulative_rewards):

        if (self.step - 1) % self.log_interval == 0:
            info1 = {"c": cumulative_rewards[-1]}
            if self.results_writer:
                self.results_writer.write_row(info1)                
        
        if self.flag:
            if (self.step - 1) % self.log_interval == 0:
                if self.step == self.max_steps - 4e5 + 100:
                    rew_step = (cumulative_rewards[-1] - cumulative_rewards[-101]) / 100
                    info3 = {"r": round(rew_step, 6)}
                    if self.results_writer:
                        self.results_writer.write_row(info3)

                elif self.step >= self.max_steps - 4e5 + 101:
                    rew_step = (cumulative_rewards[-1] - cumulative_rewards[-101]) / 100
                    info3 = {"r": round(rew_step, 6)}
                    if self.results_writer:
                        self.results_writer.write_row(info3)
                        
            if self.step ==  self.max_steps:
                for i in reversed(range(99, 0, -1)):
                    if i % self.log_interval == 0:
                        rew_step = (cumulative_rewards[-1] - cumulative_rewards[-1 - i]) / i
                        info3 = {"r": round(rew_step, 6)}
                        if self.results_writer:
                            self.results_writer.write_row(info3)

                            
        else:
            if (self.step - 1) % self.log_interval == 0:
                if self.step == 100:
                    rew_step = cumulative_rewards[-1] / 100
                    info3 = {"r": round(rew_step, 6)}
                    if self.results_writer:
                        self.results_writer.write_row(info3)

                elif self.step >= 101:
                    rew_step = (cumulative_rewards[-1] - cumulative_rewards[-101])/ 100
                    info3 = {"r": round(rew_step, 6)}
                    if self.results_writer:
                        self.results_writer.write_row(info3)

                        
            if self.step ==  self.max_steps:
                for i in reversed(range(99, 0, -1)):
                    if i % self.log_interval == 0:
                        rew_step = (cumulative_rewards[-1] - cumulative_rewards[-1 - i]) / i
                        info3 = {"r": round(rew_step, 6)}
                        if self.results_writer:
                            self.results_writer.write_row(info3)                                        
                            
        self.step = self.step + 1


    def insert_entropy_rate(self, entropy_rate):
        
        if self.step - 1 >= self.max_steps - 4e5:
            info2 = {"e": entropy_rate.item()}
            if self.results_writer:
                self.results_writer.write_row(info2)

    def close(self) -> None:
        if self.results_writer is not None:
            self.results_writer.close()


class LoadMonitorResultsError(Exception):
    pass


class ResultsWriter:

    def __init__(
        self,
        filename: str = "",
        header: Optional[Dict[str, Union[float, str]]] = None
    ):
        if header is None:
            header = {}
        if not (filename.endswith(storage.EXT1) or
                filename.endswith(storage.EXT2) or
                filename.endswith(storage.EXT3)):
            if os.path.isdir(filename):
                filename1 = os.path.join(filename, storage.EXT1)
                filename2 = os.path.join(filename, storage.EXT2)
                filename3 = os.path.join(filename, storage.EXT3)
            else:
                filename1 = filename + "." + storage.EXT1
                filename2 = filename + "." + storage.EXT2
                filename3 = filename + "." + storage.EXT3
                
        # Prevent newline issue on Windows, see GH issue #692
        self.file_handler1 = open(filename1, "wt", newline="\n")
        self.file_handler2 = open(filename2, "wt", newline="\n")
        self.file_handler3 = open(filename3, "wt", newline="\n")

        self.file_handler1.write("#%s\n" % json.dumps(header))
        self.file_handler2.write("#%s\n" % json.dumps(header))
        self.file_handler3.write("#%s\n" % json.dumps(header))
        
        self.logger = (csv.DictWriter(self.file_handler1, fieldnames=('c')),
                       csv.DictWriter(self.file_handler2, fieldnames=('r')),
                       csv.DictWriter(self.file_handler3, fieldnames=('e')))
        
        self.logger[0].writeheader()
        self.logger[1].writeheader()        
        self.logger[2].writeheader()
        
        self.file_handler1.flush()
        self.file_handler2.flush()
        self.file_handler3.flush()
        

    def write_row(self, epinfo: Dict[str, Union[float, int]]) -> None:

        if self.logger:
            if list(epinfo.keys())[0] == 'c':
                self.logger[0].writerow(epinfo)
                self.file_handler1.flush()
            elif list(epinfo.keys())[0] == 'r':
                self.logger[1].writerow(epinfo)
                self.file_handler2.flush()
            elif list(epinfo.keys())[0] == 'e':
                self.logger[2].writerow(epinfo)            
                self.file_handler3.flush()
            

    def close(self) -> None:
        self.file_handler1.close()
        self.file_handler2.close()
        self.file_handler3.close()
        


def get_monitor_files(path: str) -> Tuple[str]:
    return (glob(os.path.join(path, "*" + storage.EXT1))[0],
            glob(os.path.join(path, "*" + storage.EXT2))[0],
            glob(os.path.join(path, "*" + storage.EXT3))[0])


def load_results(path: str) -> pandas.DataFrame:
    monitor_files = get_monitor_files(path)
    if len(monitor_files) == 0:
        raise LoadMonitorResultsError(f"No monitor files of the form *{storage.EXT} found in {path}")
    data_frames =  []
    for file_name in monitor_files:
        with open(file_name) as file_handler:
            first_line = file_handler.readline()
            assert first_line[0] == "#"
            data_frame = pandas.read_csv(file_handler, index_col=None)
        data_frames.append(data_frame)  
    return data_frames


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
