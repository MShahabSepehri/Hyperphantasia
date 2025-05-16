import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from utils import io_tools
from models.answering import ANSWERING_CLASS_DICT, DEFAULT_MODEL_CONFIGS 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='gpt')
    parser.add_argument("--model_args_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default='shahab7899/Hyperphantasia')
    parser.add_argument("--difficulty", type=str, default=None, choices=[None, 'easy', 'medium', 'hard'])
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--redo_invalids", default=False, action='store_true')
    parser.add_argument("--local_data", default=False, action='store_true')
    args = parser.parse_args()

    if not args.local_data and args.difficulty is None:
        raise ValueError("Please provide a difficulty for using Huggingface dataset.")

    if args.model_args_path is None:
        args.model_args_path = DEFAULT_MODEL_CONFIGS.get(args.model_name)

    if args.dataset_name is None:
        if args.difficulty is not None:
            args.dataset_name = args.difficulty
        else:
            tmp = args.dataset_path
            tmp = tmp.replace('//', '/')
            tmp = tmp.split('/')
            if len(tmp[-1]) > 0:
                args.dataset_name = tmp[-1]
            else:
                args.dataset_name = tmp[-2]

    return args

if __name__ == "__main__":
    args = get_args()
    ROOT = io_tools.get_root(__file__, 2)

    save_path = f'{ROOT}/Results/'
    if args.config is None:
        config = {}
    else:
        config = io_tools.load_json(args.config)
    answering_class = ANSWERING_CLASS_DICT.get(args.model_name)
    ans_obj = answering_class(model_args_path=args.model_args_path, 
                              dataset_path=args.dataset_path, 
                              dataset_name=args.dataset_name,
                              device=args.device, 
                              difficulty=args.difficulty, 
                              redo_invalids=args.redo_invalids,
                              local_data=args.local_data,
                              **config)
    ans_obj.evaluate(args.resume_path, save_path)