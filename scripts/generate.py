import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import random
import shutil
import argparse
import numpy as np
from utils import io_tools
from puzzles import seven_seg, connect_the_dots, ball_trajectory, triangle_completion

PUZZLE_DICT = {
    "seven_seg": seven_seg,
    "connect_the_dots": connect_the_dots,
    "ball_trajectory": ball_trajectory,
    "triangle_completion": triangle_completion,
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    ROOT = io_tools.get_root(__file__, 2)
    np.random.seed(args.seed)
    random.seed(args.seed)

    generation_config = io_tools.load_json(args.config)
    image_dir = generation_config.get("image_dir")
    image_format = generation_config.get("image_format")
    image_size = tuple(generation_config.get("image_size"))
    save_dir = generation_config.get("save_dir")
    config_dict = generation_config.get('configs')
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)

    dataset = {}
    meta_data = {}
    meta_data["Image_dir"] = image_dir
    meta_data["Image_format"] = image_format
    for key in config_dict.keys():
        print(key)
        config = io_tools.load_json(key)
        num_samples = config_dict.get(key)
        puzzle_lib = PUZZLE_DICT.get(config.get('puzzle'))
        # raise ValueError(config)
        tmp = puzzle_lib.create(**config.get("params"), 
                                image_dir=image_dir,
                                image_format=image_format, 
                                num_samples=num_samples,
                                image_size=image_size)
        dataset.update(tmp)

        meta_data[key] = {"config": config, "num_samples": num_samples}

    io_tools.save_json(dataset, f'{save_dir}/dataset.json')
    io_tools.save_json(meta_data, f'{save_dir}/meta_data.json')