import os
import puzzles.image_utils as iu
import numpy as np
from utils import io_tools
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

ROOT = io_tools.get_root(__file__, 2)
TEMPLATE = io_tools.load_json(f'{ROOT}/configs/templates/questions.json').get('seven_seg')

EDGE_DICT = {
    0: [(1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (3, 6)],
    1: [(4, 5), (5, 6)],
    2: [(2, 3), (1, 4), (4, 5), (3, 6), (2, 5)],
    3: [(1, 4), (4, 5), (5, 6), (3, 6), (2, 5)],
    4: [(1, 2), (4, 5), (5, 6), (2, 5)],
    5: [(1, 4), (1, 2), (5, 6), (3, 6), (2, 5)],
    6: [(1, 2), (2, 3), (1, 4), (5, 6), (3, 6), (2, 5)],
    7: [(4, 5), (5, 6), (1, 4)],
    8: [(1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (3, 6), (2, 5)],
    9: [(1, 2), (1, 4), (4, 5), (5, 6), (3, 6), (2, 5)],
}


def create_dots(n=3, do_shuffle=True):
    dots = list(range(6*n))
    loc_dict = {}
    if do_shuffle:
        shuffle(dots)
    for i in range(2*n):
        for j in range(3):
            loc_dict[dots[i*3+j]] = (j, i)
    return loc_dict, dots

def create_numbers(n=3, do_shuffle=True, numbers=None):
    edges = []
    if numbers is None:
        numbers = [int(x) for x in np.random.randint(0, 10, (n))]
    loc_dict, dots = create_dots(n=n, do_shuffle=do_shuffle)
    for i in range(n):
        labels = dots[i*6:(i+1)*6]
        edge_list = EDGE_DICT.get(numbers[i])
        for e in edge_list:
            edges.append((labels[e[0]-1], labels[e[1]-1]))
    return loc_dict, edges, numbers


def generate_id(num, length=5, difficulty='M'):
    tmp = '{:0' + str(length) + 'd}'
    tmp = tmp.format(num)
    ID = f'Int_7Seg_{difficulty}_{tmp}'
    return ID

def create(n, 
           image_dir,
           sort_nodes=True, 
           sort_edges=True,
           image_format='jpg', 
           pad_inches=0.2, 
           r=0.03,
           b=0.1, 
           dot_color='black', 
           num_color='red', 
           edge_color='blue',
           fontsize=10, 
           quality=None, 
           num_samples=1,
           ordered=False,
           id_length=5,
           difficulty='M',
           image_size=(384, 384),
           dpi=300):
    
    dataset = {}
    io_tools.check_and_create_dir(image_dir)

    number_list = list(range(10 ** n))
    if not ordered:
        shuffle(number_list)
    number_list = number_list[:num_samples]
    counter = 0
    for i in tqdm(number_list):
        tmp = '{:0' + str(n) + 'd}'
        numbers = [int(x) for x in tmp.format(i)]
        loc_dict, edges, numbers = create_numbers(n=n, do_shuffle=(not sort_nodes), numbers=numbers)

        if sort_nodes:
            puzzle_path = f'{image_dir}/Seven_segment_{n}.{image_format}'
        else:
            puzzle_path = f'{image_dir}/{ID}.{image_format}'
        plot_dots(n, loc_dict, save_path=puzzle_path, edges=None, scale=1, r=r, b=b, dot_color=dot_color, 
                  num_color=num_color, fontsize=fontsize, quality=quality, pad_inches=pad_inches, 
                  image_format=image_format, dpi=dpi, image_size=image_size)

        ID = generate_id(counter, length=id_length, difficulty=difficulty)
        solution_path = f'{image_dir}/solution_{ID}.{image_format}'
        plot_dots(n, loc_dict, save_path=solution_path, edges=edges, scale=1, r=r, b=b, dot_color=dot_color, 
                  num_color=num_color, fontsize=fontsize, quality=quality, pad_inches=pad_inches, 
                  image_format=image_format, edge_color=edge_color, dpi=dpi, image_size=image_size)
        
        if sort_edges:
            edges = [(x[0], x[1]) if x[0] < x[1] else (x[1], x[0]) for x in edges]
            edges = sorted(edges, key=lambda x: x[0])
        else:
            shuffle(edges)
        connections = ', '.join([f'{x} to {y}' for (x, y) in edges])
        dataset[ID] = {
            'ID': ID,
            'Question': TEMPLATE.format(connections, n),
            'Image_file': puzzle_path.split('/')[-1],
            'Pseudo_solution_file': solution_path.split('/')[-1],
            'Answer': ''.join([str(x) for x in numbers]),
            'Category': 'Interpolation-7seg',
            'Difficulty': difficulty,
        }
        counter += 1
    return dataset

def edge_2_loc(edges, loc_dict):
    results = []
    for e in edges:
        loc1 = loc_dict.get(e[0])
        loc2 = loc_dict.get(e[1])
        results.append((loc1, loc2))
    return results

def plot_circle(ax, loc, n, radius=0.2, color='white', edgecolor='black', linewidth=1):
    y, x = loc
    y = n - 1 - y
    circle = plt.Circle((x, y), radius, color=color, ec=edgecolor, linewidth=linewidth)
    ax.add_patch(circle)

def plot_dots(n, loc_dict, save_path, edges=None, scale=1, r=0.03, b=0.1, dot_color='black', num_color='red', image_size=(384, 384),
              edge_color='blue', fontsize=10, quality=None, pad_inches=0.4, image_format='jpg', lw=1, dpi=300):
    columns = n * 2
    rows = 3
    fig, ax = plt.subplots(figsize=(scale * columns, scale * rows))  
    plt.rcParams['font.weight'] = 'bold'

    ax.set_xticks(range(columns))
    ax.set_xticks([x - 0.5 for x in range(columns)], minor=True) 
    ax.set_yticks(range(rows))  
    ax.set_yticks([x - 0.5 for x in range(rows)], minor=True)  
        
    # Create grid  
    ax.spines[['right', 'left', 'bottom', 'top']].set_visible(False)

    # Turn off the major ticks  
    ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)  
    ax.tick_params(which='minor', bottom=False, left=False, labelbottom=False, labelleft=False)
    for num in loc_dict.keys():
        loc = loc_dict.get(num)
        plot_circle(ax, (loc[0]+b, loc[1]-b), n=rows, radius=r, color=dot_color)
        ax.text(loc[1], rows-1-loc[0], num, ha='center', va='center', fontsize=fontsize, color=num_color) 
    
    if edges is not None:
        for edge in edge_2_loc(edges, loc_dict):
            loc1, loc2 = edge
            ax.plot([loc1[1]-b, loc2[1]-b], [rows-loc1[0]-1-b, rows-loc2[0]-1-b], color=edge_color, linewidth=lw)

    if quality is not None:
        tmp_save_path = f'tmp_puzzle_image.{image_format}'
        plt.savefig(tmp_save_path, format=image_format, bbox_inches='tight', pad_inches=pad_inches, pil_kwargs={"quality": quality})
        iu.save_square_image(tmp_save_path, save_path, output_size=image_size, format=image_format)
        os.remove(tmp_save_path)
    else:
        tmp_save_path = f'tmp_puzzle_image.{image_format}'
        plt.savefig(tmp_save_path, format=image_format, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)
        iu.save_square_image(tmp_save_path, save_path, output_size=image_size, format=image_format)
        os.remove(tmp_save_path)
    plt.close()
