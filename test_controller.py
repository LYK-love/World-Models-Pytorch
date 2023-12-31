""" Test controller """
import argparse
from os.path import join, exists

# gym==0.9.4
from utils.misc import RolloutGenerator

# gym==0.26.2
# from utils.misc_for_v2 import RolloutGenerator

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Where models are stored.')

    # Add the 'render' argument
    parser.add_argument('--render', action='store_true', help='Enable rendering.')

    args = parser.parse_args()

    ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

    assert exists(ctrl_file), \
        "Controller was not trained..."

    device = torch.device('cpu')

    generator = RolloutGenerator(args.logdir, device, 1000)

    with torch.no_grad():
        neg_cumulative = generator.rollout(None, render=args.render)
        print(neg_cumulative)

