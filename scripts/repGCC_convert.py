import torch
import argparse

import os, sys
# get the project root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.modules.gcc_cvx_modules import convert_state_dict

def get_args_parser():
    parser = argparse.ArgumentParser('RepGCC checkpoint to normal GCC checkpoint', add_help=False)
    parser.add_argument('--source_path', default='', type=str, help='path to RepGCC checkpoint')
    parser.add_argument('--target_path', default='', type=str, help='path to normal GCC checkpoint')
    return parser

def main(args):
    old_checkpoint = torch.load(args.source_path)
    print("old checkpoint file loaded")
    old_state_dict = old_checkpoint['state_dict']
    new_state_dict = convert_state_dict(old_state_dict)
    print("state_dict convert done")
    new_checkpoint = old_checkpoint
    new_checkpoint['state_dict'] = new_state_dict
    torch.save(new_checkpoint, args.target_path)
    print("new checkpoint file saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RepGCC checkpoint to normal GCC checkpoint', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)