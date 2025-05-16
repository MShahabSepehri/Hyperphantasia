import numpy as np
import matplotlib.pyplot as plt
from utils import io_tools
from tqdm import tqdm

ROOT = io_tools.get_root(__file__, 2)
TEMPLATE = io_tools.load_json(f'{ROOT}/configs/templates/questions.json').get('triangle_completion')

CHOICES = ["A", "B", "C", "D"]
NUM_CHOICES = len(CHOICES)

def create(image_dir,
           image_size,
           revealed_trajectory_percentage,
           image_format='jpg', 
           dot_color='black', 
           line_color="blue",
           dpi=64,
           num_samples=1,
           id_length=5,
           difficulty='M'):
    
    dataset = {}
    io_tools.check_and_create_dir(image_dir)

    for i in tqdm(range(num_samples)):
        # Generate a unique ID for the puzzle
        ID = f'Ext_Tri_{difficulty}_{str(i).zfill(id_length)}'
        
        # Generate paths
        puzzle_path = f'{image_dir}/{ID}.{image_format}'
        solution_path = f'{image_dir}/solution_{ID}.{image_format}'
        
        # Draw coordinates randomly
        sign = np.random.choice([-1, 1])
        intersect_y = sign * (np.random.beta(2,1) + 1) / 2 # Between 0.5 and 1
        intersect_x = np.random.uniform(-0.5, 1)
        choice = np.random.choice(range(NUM_CHOICES))

        # [1, intersect_y] is the coordinate of the choice "A". Calculate the coordinates of the other choices based on intersect_x
        coordinates = []
        for i in range(len(CHOICES)):
            if CHOICES[i] == "A":
                coordinates.append((1, intersect_y))
            else:
                coordinates.append((1 - (i/(2*(NUM_CHOICES-1))) * (1-intersect_x), intersect_y - (i/(2*(NUM_CHOICES-1))) * intersect_y))
                        
        # Save the puzzle image
        plt.figure(figsize=(image_size[0]//dpi, image_size[1]//dpi))
        
        # Draw a line between [-1,0] and [intersect_x, 0]
        x2 = np.linspace(-1, intersect_x, 100)
        y2 = (0 - 0) / (intersect_x - (-1)) * (x2 - (-1)) + 0
        plt.plot(x2, y2, c=line_color, linewidth=5)

        # Draw a line between [intersect_x, 0] and [1, intersect_y] 
        x3 = np.linspace(intersect_x, 1, 100)
        y3 = (intersect_y - 0) / (1 - intersect_x) * (x3 - intersect_x)
        plt.plot(x3, y3, c=line_color, linewidth=5)

        # Draw a line between [-1,0] and [coordinate[0], coordinate[1]]
        x = np.linspace(-1, (coordinates[choice][0] + 1) * revealed_trajectory_percentage - 1, 100)
        y = (coordinates[choice][1]) / (coordinates[choice][0] + 1) * (x + 1)
        plt.plot(x, y, c='red', linewidth=5)
        plt.plot(-1, 0, marker='o', markersize=15, color='red')  # Draw the left point

        # Draw the choices with marker "o" and also put letters A, B, C, D at the coordinates on top of the markers
        for i, (x_coord, y_coord) in enumerate(coordinates):
            plt.plot(x_coord, y_coord, marker='o', markersize=15, color=dot_color)
            if sign == 1:
                plt.text(x_coord + 0.06, y_coord - 0.06, CHOICES[i], fontsize=18, ha='left', va='bottom', color='black')
            else:
                plt.text(x_coord + 0.06, y_coord + 0.06, CHOICES[i], fontsize=18, ha='left', va='top', color='black')

        # Set the limits and labels
        delta = (1.0 - abs(intersect_y))/2
        plt.xlim(-1.1, 1.1)
        if sign == 1:
            plt.ylim(-0.1-delta, 1.1 - delta)
        else:   
            plt.ylim(-1.1 + delta, 0.1+delta)

        # Turn off the axes
        plt.axis('off')
        # Save the puzzle image
        plt.savefig(puzzle_path, format=image_format, dpi=dpi)
        
        # Complete the line and save the solution image
        x = np.linspace((coordinates[choice][0] + 1) * revealed_trajectory_percentage - 1, coordinates[choice][0], 100)
        y = (coordinates[choice][1]) / (coordinates[choice][0] + 1) * (x + 1)
        plt.plot(x, y, ":", c='red', linewidth=5)
        plt.savefig(solution_path, format=image_format, dpi=dpi)
        plt.close()
        
        dataset[ID] = {
            'ID': ID,
            'Question': TEMPLATE,
            'Image_file': puzzle_path.split('/')[-1],
            'Pseudo_solution_file': solution_path.split('/')[-1],
            'Answer': CHOICES[choice],
            'Category': 'Extrapolation-TriangleCompletion',
            "Difficulty": difficulty,
        }
        
    return dataset