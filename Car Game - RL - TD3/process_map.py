import numpy as np
from random import random, randint
from PIL import Image as PILImage

# 1429, 660

def get_random_road_point():
    while True:
        # Randomly select a point within the map boundaries
        x = randint(0, sand.shape[0] - 1)
        y = randint(0, sand.shape[1] - 1)

        # Check if the selected point is on the road (sand value of 0)
        if sand[x, y] == 0:
            return x, y
        

#fixed_destinations = [get_random_road_point() for _ in range(4)]
#print(fixed_destinations)

def compute_obstacle_field(obstacle_influence_radius=20):
    """
    Compute an obstacle field where each vector points away from the nearest obstacle.

    Args:
        sand: 2D numpy array representing the environment (0 for roads, 1 for obstacles).
        viable_points: Array of coordinates representing viable (road) points.
        obstacle_influence_radius: Radius around each point to consider for obstacles.

    Returns:
        obstacle_field: 2D numpy array representing the obstacle field.
    """
    

    for point in viable_points:

        # Initialize obstacle vector
        obstacle_vector = np.array([0.0, 0.0])
        # Check surrounding area for obstacles
        for i in range(-obstacle_influence_radius, obstacle_influence_radius + 1):
            for j in range(-obstacle_influence_radius, obstacle_influence_radius + 1):
                check_point = point + np.array([i, j])

                # Ensure the check point is within the bounds of the map
                if 0 <= check_point[0] < sand.shape[0] and 0 <= check_point[1] < sand.shape[1]:
                    if sand[check_point[0], check_point[1]] == 1:  # Obstacle found
                        direction = point - check_point
                        distance = np.linalg.norm(direction)
                        normalized_direction = direction / distance if distance != 0 else direction

                        # Weight vector by inverse of distance (closer obstacles have more influence)
                        obstacle_vector += normalized_direction / (distance**2 if distance != 0 else 1)

        # Normalize the obstacle vector
        norm = np.linalg.norm(obstacle_vector)
        if norm != 0:
            obstacle_vector /= norm

        if np.isnan(obstacle_vector).any():
            obstacle_vector = np.array([0.0, 0.0])

        sand_field[point[0], point[1]] = obstacle_vector
        #print(obstacle_vector)


def calculate_vector_field_destination(destination): # weight_dest=0.7
    """
    Calculates a weighted vector field considering both the destination and obstacles.

    Args:
        destination: Coordinates of the destination point (x, y).
        obstacle_field: 2D array representing the vector field for obstacles.

    Returns:
        vector: Weighted vector for the car's location, or None for non-road locations.
    """

    vector_field = np.empty((sand.shape[0], sand.shape[1], 2))
    vector_field[:] = np.nan

    # Calculate vectors towards the destination for viable points
    for point in viable_points:
        direction_to_dest = destination - point
        direction_to_dest_norm = np.linalg.norm(direction_to_dest)

        if not direction_to_dest_norm:
            norm_direction = np.array([0.0, 0.0])
        else:
            norm_direction = direction_to_dest / direction_to_dest_norm

        vector_field[point[0], point[1]] = norm_direction    
    return(vector_field)

fixed_destinations = [(1197, 283), (462, 334), (1055, 556), (648, 349)]

img = PILImage.open("./images/mask.png").convert('L')
sand = np.asarray(img)/255

# Initialize vector field
viable_points = np.argwhere(sand == 0)
sand_field = np.empty((sand.shape[0], sand.shape[1], 2))
sand_field[:] = np.nan

for i, dest in enumerate(fixed_destinations):
    vector_field = calculate_vector_field_destination(np.array(dest))
    np.save(f'fields/vectorfield_{i}', vector_field)
    print(np.nansum(vector_field))
    print(f"saved vector field {i}")

compute_obstacle_field(obstacle_influence_radius=5)
print(f"Compute SandField")
np.save('fields/sandfield', sand_field)
print(np.nansum(sand_field))
print(f"saved Sand field")