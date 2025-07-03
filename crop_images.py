import cv2
import numpy as np
import os

def load_image(image_path):
    return cv2.imread(image_path)

# Convert image to grayscale and apply Gaussian blur
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 1.2)
    return blurred_image

# Apply Laplacian filter and threshold result to detect edges
def laplacian_edge_detection(blurred_image, threshold):
    laplacian_kernel = np.array([[1, 4, 1], [4, -20, 4], [1, 4, 1]])
    result = cv2.filter2D(blurred_image, -1, laplacian_kernel)
    _, thresh = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)
    return thresh

# Compute convex hull around all contour points
def convex_hull(contours):
    all_points = [pt[0] for c in contours for pt in c]
    if all_points:
        return cv2.convexHull(np.array(all_points))
    return None

# Find contours from thresholded edges and crop image to bounding rect of hull
def crop_image(thresh, image):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    points = convex_hull(contours)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        return image[y:y+h, x:x+w]
    return image 

def main():
    input_folder = "voynich_images"
    output_folder = "cropped_voynich_images"
    threshold_l = 190

    os.makedirs(output_folder, exist_ok=True)

    for i in range(3, 207):
        image_name = f"page_{i}.jpg"
        image_path = os.path.join(input_folder, image_name)

        print(f"Processing {image_name}...")

        image = load_image(image_path)
        if image is None:
            print(f"Could not read {image_path}. Skipping.")
            continue

        blurred = preprocess_image(image)
        thresh = laplacian_edge_detection(blurred, threshold_l)
        cropped_image = crop_image(thresh, image)

        output_path = os.path.join(output_folder, f"page_{i}_cropped.jpg")
        cv2.imwrite(output_path, cropped_image)
        print(f"Saved cropped image to {output_path}")

if __name__ == "__main__":
    main()
