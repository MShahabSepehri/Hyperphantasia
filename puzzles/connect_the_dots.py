import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from utils import io_tools
import matplotlib.pyplot as plt
import puzzles.image_utils as iu

ROOT = io_tools.get_root(__file__, 2)
TEMPLATE = io_tools.load_json(f'{ROOT}/configs/templates/questions.json').get('connect_the_dots')
LETTER_OPTIONS = ['A', 'B', 'C', 'D']

def find_corners(image_path, kernel_size=3, minDistance=5, image_size=(384, 384)):
    img = cv2.imread(image_path)
    img = iu.pad_to_square(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, image_size)
    minDistance = int(max(gray.shape[:2]) / 360 * minDistance)

    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = np.zeros_like(gray)
    cv2.drawContours(result, contours, -1, 255, thickness=1)

    if kernel_size > 0:
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
    result = (result > 64) * (np.zeros_like(result) + 255)

    points = [(y, x) for (x, y) in zip(*np.where(result > 0))]

    points = np.asarray(points)
    
    remaining = points.copy()
    selected = []

    while len(remaining) > 0:
        # Select the first point and remove nearby ones
        point = remaining[0]
        selected.append(tuple(point))
        dists = np.linalg.norm(remaining - point, axis=1)
        remaining = remaining[dists > minDistance]


    random.shuffle(selected)

    return selected, result


def is_connected(binary_image, point1, point2, box_width=6):
    """
    Check if two points are connected by a line in a binary image.
    
    Args:
        binary_image (np.ndarray): Binary image (edges = 255 or 1, background = 0)
        point1 (tuple): (x1, y1)
        point2 (tuple): (x2, y2)
        threshold (float): Percentage of points that must be on a line to consider connected
        
    Returns:
        bool: True if connected, False otherwise
    """
    x1, y1 = point1
    x2, y2 = point2

    # Generate line points between the two points
    num_samples = int(np.hypot(x2 - x1, y2 - y1))  # number of samples based on distance
    if num_samples == 0:
        return False

    x_vals = np.linspace(x1, x2, num_samples).astype(int)
    y_vals = np.linspace(y1, y2, num_samples).astype(int)
    
    # Clip coordinates to be inside the image
    x_vals = np.clip(x_vals, 0, binary_image.shape[1]-1)
    y_vals = np.clip(y_vals, 0, binary_image.shape[0]-1)

    hits = 0
    total = 0

    half_box = box_width // 2

    for x, y in zip(x_vals, y_vals):
        # Define bounding box around each sampled point
        x_min = max(x - half_box, 0)
        x_max = min(x + half_box + 1, binary_image.shape[1])
        y_min = max(y - half_box, 0)
        y_max = min(y + half_box + 1, binary_image.shape[0])

        patch = binary_image[y_min:y_max, x_min:x_max]
        
        total += patch.size
        hits += np.count_nonzero(patch)

    connection_ratio = hits / total
    return connection_ratio


def find_connections(image, point, point_list, search_step=6, dist_tr=6, search_tr=50):
    search_list = [point]
    visited = [point]
    connections = []
    while len(search_list) > 0:
        p = search_list.pop(0)
        new_connectoins = []
        for i in range(-search_step, search_step + 1):
            for j in range(-search_step, search_step + 1):
                x, y = (p[0] + i, p[1] + j)
                if x < 0 or y < 0:
                    continue
                if x >= image.shape[1] or y >= image.shape[0]:
                    continue
                if image[y, x] < 255:
                    continue
                if (x, y) in visited:
                    continue
                if (x, y) in point_list:
                    new_connectoins.append((x, y))
                else:
                    search_list.append((x, y))
        visited += search_list
        connections += new_connectoins
        for p in new_connectoins:
            search_list = [x for x in search_list if (abs(x[0] - p[0]) + abs(x[1] - p[1]) > dist_tr)]
            search_list = [x for x in search_list if (abs(x[0] - point[0]) + abs(x[1] - point[1]) < search_tr)]
    return connections

def image_to_puzzle(image_path,
                    minDistance=5, 
                    kernel_size=5, 
                    search_step=4, 
                    dist_tr=10,
                    search_tr=50,
                    image_size=(384, 384)):
    corners, image = find_corners(image_path,  
                                  minDistance=minDistance, 
                                  kernel_size=kernel_size,
                                  image_size=image_size)
    edges = []
    edges = find_edges(image, corners, 
                       search_step=search_step, 
                       dist_tr=dist_tr, 
                       search_tr=search_tr)
    loc_dict = {i: corners[i] for i in range(len(corners))}
    return image, loc_dict, edges
    
def find_edges(image, corners, search_step=4, dist_tr=10, search_tr=50):
    edges = []
    for i, c in enumerate(corners):
        connections = find_connections(image, c, corners, 
                                       search_step=search_step, 
                                       dist_tr=dist_tr,
                                       search_tr=search_tr)
        connections = [corners.index(c) for c in connections]
        edges += [(i, c) if i < c else (c, i) for c in connections]
    return list(set(edges))

def create_puzzle_images(image, 
                         loc_dict, 
                         edges, 
                         save_path,
                         image_size=(384, 384), 
                         hide_image=False, 
                         r=20, 
                         b=1, 
                         dot_color='red', 
                         text_color='black', 
                         edge_color='blue',
                         image_format='jpg', 
                         fontsize=10, 
                         pad_inches=0,
                         dpi=300,
                         lw=1):
    _, ax = plt.subplots(figsize=(8, 8))
    if hide_image:
        tmp = image.copy() * 0 + 255
        ax.imshow(tmp, cmap='gray', vmin=0, vmax=255)
    else:
        ax.imshow(255 - image, cmap='gray', vmin=0, vmax=255)
    ax.scatter([c[0] for c in loc_dict.values()], [c[1] for c in loc_dict.values()], c=dot_color, s=r) 
    if fontsize > 0:
        for key in loc_dict.keys():
            loc = loc_dict.get(key)
            ax.text(loc[0] + b, loc[1] - b, key, ha='center', va='center', fontsize=fontsize, color=text_color) 
    for e in edges:
            p1 = loc_dict[e[0]]
            p2 = loc_dict[e[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_color, linewidth=lw)
    ax.axis('off')
    
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi, format=image_format)
    tmp_save_path = f'tmp_puzzle_image.{image_format}'
    plt.savefig(tmp_save_path, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi, format=image_format)
    iu.save_square_image(tmp_save_path, save_path, output_size=image_size, format=image_format)
    os.remove(tmp_save_path)
    plt.close()

def generate_id(counter, length=5, difficulty='M'):
    tmp = '{:0' + str(length) + 'd}'
    tmp = tmp.format(counter)
    ID = f'Int_Connect_{difficulty}_{tmp}'
    return ID

def create(data_path, 
           image_dir,
           image_format='jpg', 
           minDistance=10,
           kernel_size=0,
           search_step=8,
           dist_tr=15,
           search_tr=50,
           dot_color='black', 
           num_color='red', 
           edge_color='blue',
           fontsize=10, 
           dpi=300,
           num_samples=1,
           r=0.03,
           b=0.1, 
           lw=1,
           id_length=5,
           image_size=(384, 384),
           difficulty='M'):

    counter = 0
    image_dataset = io_tools.load_json(data_path)
    dataset = {}
    all_labels = list(set([x.get("label") for x in image_dataset.values()]))
    key_list = list(image_dataset.keys())[:num_samples]
    for key in tqdm(key_list):
        sample = image_dataset.get(key)
        label = sample.get("label")
        image_path = sample.get("image_path")
        options = sample.get("similar_labels")

        image, loc_dict, edges = image_to_puzzle(image_path,
                                                 minDistance=minDistance, 
                                                 kernel_size=kernel_size, 
                                                 search_step=search_step, 
                                                 dist_tr=dist_tr,
                                                 search_tr=search_tr,
                                                 image_size=image_size
                                                 )
        ID = generate_id(counter, id_length, difficulty)

        puzzle_path = f'{image_dir}/{ID}.{image_format}'
        create_puzzle_images(image, loc_dict, edges=[], hide_image=True, r=r, b=b, lw=lw, dpi=dpi,
                             image_size=image_size, fontsize=fontsize, save_path=puzzle_path,
                             dot_color=dot_color, edge_color=edge_color, text_color=num_color, 
                             image_format=image_format)

        solution_path = f'{image_dir}/solution_{ID}.{image_format}'
        create_puzzle_images(image, loc_dict, edges=[], hide_image=False, r=r, b=b, lw=lw, dpi=dpi,
                             image_size=image_size, fontsize=fontsize, save_path=solution_path,
                             dot_color=dot_color, edge_color=edge_color, text_color=num_color, 
                             image_format=image_format)

        connections = ', '.join([f'{x} to {y}' for (x, y) in edges])
        options, answer = generate_options(all_labels, label, options)
        dataset[ID] = {
            'ID': ID,
            # 'Question': TEMPLATE.format(connections, *options),
            'Question': TEMPLATE.format(*options),
            'Image_file': puzzle_path.split('/')[-1],
            'Pseudo_solution_file': solution_path.split('/')[-1],
            'Answer': answer,
            'Category': 'Interpolation-connect',
            'Difficulty': difficulty,
        }
        counter += 1
    return dataset

def generate_options(all_labels, label, options=None):
    if options is not None:
        options.append(label)
        random.shuffle(options)
        return options, LETTER_OPTIONS[options.index(label)] 
    tmp = [x for x in all_labels if x != label]
    options = random.sample(tmp, 3)
    options.append(label)
    random.shuffle(options)
    return options, LETTER_OPTIONS[options.index(label)]