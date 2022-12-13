import torch 
import re
from typing import List, Optional, Tuple, Union
import os
import json
import glob 
import matplotlib.pyplot as plt

import click

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


@click.command()
@click.option('--experiment', 'experiment_folder', help='Experiment folder name', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
def evaluate_experiment(
    experiment_folder: str,
    truncation_psi: float,
):
    # get info from training options
    training_options_path = os.path.join(experiment_folder, "training_options.json")
    with open(training_options_path, "r") as f:
        training_options = json.load(f)

    # train psi-truncation
    psi_train = training_options["pose_trunc_dist"]

    # learning rate
    lr = training_options["T_opt_kwargs"]["lr"]

    print(psi_train, lr)

    # iterate over epochs

    # scan pkls in folder 
    pkls = [f for f in glob.glob(os.path.join(experiment_folder, "*.pkl"))]
    pkls.sort()

    # kimgs
    kimgs = [int(name[-8:-4]) for name in pkls]

    experiment_name = os.path.basename(experiment_folder)
    print(experiment_name)

    # generate imgs over designated psi-truncations
    for pkl, kimg in zip(pkls, kimgs):
        pkl_folder = os.path.join("./out/", f"{experiment_name}_psi_{truncation_psi}", f"kimg_{kimg}")

        os.system(f"python gen_images.py --outdir={pkl_folder} --trunc={truncation_psi} --seeds=0-99 --network={pkl}")


    
    # evaluate over imgs





if __name__ == "__main__":
    evaluate_experiment()