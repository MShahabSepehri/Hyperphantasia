import numpy as np
import matplotlib.pyplot as plt
from utils import io_tools
from tqdm import tqdm

ROOT = io_tools.get_root(__file__, 2)
TEMPLATE = io_tools.load_json(f'{ROOT}/configs/templates/questions.json').get('ball_trajectory')

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
        ID = f'Ext_Ball_{difficulty}_{str(i).zfill(id_length)}'
        
        # Generate paths
        puzzle_path = f'{image_dir}/{ID}.{image_format}'
        solution_path = f'{image_dir}/solution_{ID}.{image_format}'
        
        # Draw coordinates randomly
        left_x = np.random.uniform(-1, 0)
        top_y = np.random.uniform(0.25, 1) # Between 0.5 and 1
        choice = np.random.choice(range(NUM_CHOICES))

        # [1, intersect_y] is the coordinate of the choice "A". Calculate the coordinates of the other choices based on intersect_x
        coordinates = []
        for i in range(len(CHOICES)):
            if CHOICES[i] == "A":
                coordinates.append((1, 0))
            else:
                coordinates.append((1 - i/(2*(NUM_CHOICES-1)), 0))
                
        # Save the puzzle image
        plt.figure(figsize=(image_size[0]//dpi, image_size[1]//dpi))

        # Draw a line between [-1, 0] and [1, 0] 
        x3 = np.linspace(-1, 1, 100)
        plt.plot(x3, [0]*len(x3), c=line_color, linewidth=5)

        # Draw the curve that goes through [left_x,0], [(left_x + coordinates[choice][0])/2, top_y], and [coordinates[choice][0], 0]
        x = np.linspace(left_x, (coordinates[choice][0] - left_x) * revealed_trajectory_percentage + left_x, 100)
        y = -4 * top_y / (coordinates[choice][0] - left_x)**2 * (x - left_x) * (x - coordinates[choice][0])
        plt.plot(x, y, c='red', linewidth=5)
        plt.plot(left_x, 0, marker='o', markersize=15, color='red')  # Draw the left point

        # Draw the choices with marker "o" and also put letters A, B, C, D at the coordinates on top of the markers
        for i, (x_coord, y_coord) in enumerate(coordinates):
            plt.plot(x_coord, y_coord, marker='o', markersize=15, color=dot_color)
            plt.text(x_coord, y_coord - 0.05, CHOICES[i], fontsize=18, ha='center', va='top', color='black')

        # Set the limits and labels
        delta = (1.0 - top_y)/2
        plt.xlim(-1.1, 1.1)
        plt.ylim(-0.1-delta, 1.0 - delta)

        # Turn off the axes
        plt.axis('off')
        # Save the puzzle image
        plt.savefig(puzzle_path, format=image_format, dpi=dpi)
        
        # Complete the line and save the solution image
        x = np.linspace((coordinates[choice][0] - left_x) * revealed_trajectory_percentage + left_x, coordinates[choice][0], 100)
        y = -4 * top_y / (coordinates[choice][0] - left_x)**2 * (x - left_x) * (x - coordinates[choice][0])
        plt.plot(x, y, ":", c='red', linewidth=5)
        plt.savefig(solution_path, format=image_format, dpi=dpi)
        plt.close()
        
        dataset[ID] = {
            'ID': ID,
            'Question': TEMPLATE,
            'Image_file': puzzle_path.split('/')[-1],
            'Pseudo_solution_file': solution_path.split('/')[-1],
            'Answer': CHOICES[choice],
            'Category': 'Extrapolation-BallTrajectory',
            "Difficulty": difficulty,
        }
        
    return dataset