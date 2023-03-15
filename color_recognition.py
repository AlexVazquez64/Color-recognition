import cv2
import numpy as np

# Load image
image = cv2.imread('orange-sample-paint.webp')

# Convert image from BGR to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to LAB color space
lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)

# Define color ranges for different paint colors in LAB color space
color_ranges = {
    'red': [(145, 175), [128, 128]],
    'blue': [(90, 140), [-10, -50]],
    'green': [(60, 90), [-40, -10]],
    'yellow': [(90, 110), [30, 50]],
    'orange': [(120, 140), [50, 80]],
    'purple': [(90, 110), [-20, 20]],
    'pink': [(140, 165), [30, 50]]
}

# Define function to find color in image
def find_color(image, color_ranges):
    # Calculate mean color values for each channel
    mean_values = np.mean(image, axis=(0, 1))

    # Convert mean color values to LAB color space
    lab_values = cv2.cvtColor(np.uint8([[mean_values]]), cv2.COLOR_RGB2LAB).flatten()

    # Find color that best matches mean LAB values
    min_distance = float('inf')
    best_match = None
    for color_name, color_range in color_ranges.items():
        distance = np.linalg.norm(lab_values - np.array(color_range).flatten())
        if distance < min_distance:
            min_distance = distance
            best_match = color_name

    return best_match

# Find color in image
color = find_color(lab_image, color_ranges)

# Print results
print('Detected color:', color)