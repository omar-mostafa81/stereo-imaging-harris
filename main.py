import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Import and Load the image function bitmap
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Convert RGB to Grayscale function
def convert_rgb_to_gray(image):
    gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r, g, b = image[i, j]
            # Formula: I = Round(0.299R + 0.587G + 0.114B)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            gray_image[i, j] = int(gray)
    return gray_image

def calculate_gradients(image):
    """Calculate vertical and horizontal gradients of an image.
    Returns: gradients in i (x is left to right), j (y is top to bottom) direction
    """
    rows, cols = image.shape
    gradient_x = np.zeros((rows, cols), dtype=np.float32)
    gradient_y = np.zeros((rows, cols), dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Sobel operator for horizontal gradient
            gradient_x[i, j] = (
                -1 * image[i - 1, j - 1] + 1 * image[i - 1, j + 1] +
                -2 * image[i, j - 1] + 2 * image[i, j + 1] +
                -1 * image[i + 1, j - 1] + 1 * image[i + 1, j + 1]
            )
            # Sobel operator for vertical gradient
            gradient_y[i, j] = (
                -1 * image[i - 1, j - 1] + -2 * image[i - 1, j] + -1 * image[i - 1, j + 1] +
                    1 * image[i + 1, j - 1] +  2 * image[i + 1, j] +  1 * image[i + 1, j + 1]
            )

    return gradient_x, gradient_y

def calculate_harris_corners(img, gauss_k, neigh_w, response_k, response_threshold):
    """ Helper function to calculate the harris corners to be used as feature points for depth estimation
    Inputs:
        1. img: gray scale image
        2. gauss_k: kernel size for gaussian filtering
        3. neigh_w: neighborhood window w used for structural matrix calculation
    Returns:
        1. Locations of corners detected in img 
    """
    # Perform Gaussian filtering on image 
    filtered_img = cv2.GaussianBlur(img, (gauss_k, gauss_k), 0) # sending sigma = 0 lets CV2 to pick the suitable sigma based on the kernal size

    # Compute horizental and vertical gradients 
    Ix, Iy = calculate_gradients(filtered_img)
    
    # For each pixel in the image:
    # 1. Calculate local structure matrix A with the given neighborhood window W
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # build summed products inside each W×W window  
    r = neigh_w // 2
    rows, cols = img.shape
    Sxx = np.zeros_like(Ixx)
    Syy = np.zeros_like(Iyy)
    Sxy = np.zeros_like(Ixy)

    # loop over every pixel (skip borders where the window would exceed image)
    for i in range(r, rows - r):
        for j in range(r, cols - r):
            # slice the W×W neighbourhood
            win_xx = Ixx[i-r:i+r+1, j-r:j+r+1]
            win_yy = Iyy[i-r:i+r+1, j-r:j+r+1]
            win_xy = Ixy[i-r:i+r+1, j-r:j+r+1]
            Sxx[i, j] = win_xx.sum()
            Syy[i, j] = win_yy.sum()
            Sxy[i, j] = win_xy.sum()

    # Structural matrix is S = 2 * A_w(x, y)
    Sxx = 2*Sxx
    Syy = 2*Syy
    Sxy = 2*Sxy

    # 2. Compute response function R(S)
    det_S = (Sxx * Syy) - (Sxy ** 2)
    trace_S = Sxx + Syy
    R = det_S - response_k * (trace_S**2)
    
    # Choose the best candidates for corners by applying a threshold on the response function R(A) 
    R_max = R.max()
    thresh = response_threshold * R_max
    R[R < thresh] = 0

    # 3×3 non‑max suppression
    corners = []
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if R[i, j] == 0:
                continue
            local_patch = R[i-1:i+2, j-1:j+2]
            if R[i, j] == local_patch.max():
                corners.append((i, j))

    # Return the location of the best candidates and R
    return corners, R

def plot_corners(gray_left,  corners_left,
                gray_right, corners_right,
                color_left='lime',  color_right='red',
                plus_size=80, save_as=None):
    """
    Show the left and right gray‑scale images in one row with “+” markers
    on their Harris corners.
    """
    # Prepare the corner coordinates
    yL, xL = zip(*corners_left)   if corners_left  else ([], [])
    yR, xR = zip(*corners_right) if corners_right else ([], [])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
    fig.subplots_adjust(wspace=0.02)

    # Left image 
    ax[0].imshow(gray_left, cmap='gray', vmin=0, vmax=255)
    ax[0].scatter(xL, yL, c=color_left, marker='+',
                  s=plus_size, linewidths=1)
    ax[0].set_title("Left image")
    ax[0].axis('off')

    # Right image 
    ax[1].imshow(gray_right, cmap='gray', vmin=0, vmax=255)
    ax[1].scatter(xR, yR, c=color_right, marker='+',
                  s=plus_size, linewidths=1)
    ax[1].set_title("Right image")
    ax[1].axis('off')

    if save_as is not None:
        fig.savefig(save_as, bbox_inches='tight', pad_inches=0)

    plt.show()

def iter_stereo_pairs(base_dir="examples"):
    """
    Generator that yields (folder_name, left_img, right_img) where the images are in grey scale.

    It assumes every immediate sub‑folder of `base_dir` contains exactly
    one file that matches *_left.png and one that matches *_right.png.
    """
    for sub in sorted(os.listdir(base_dir)):
        sub_path = os.path.join(base_dir, sub)
        # Find the left & right image paths
        left_path  = None
        right_path = None
        for fname in os.listdir(sub_path):
            if fname.endswith("_left.bmp"):
                left_path  = os.path.join(sub_path, fname)
            elif fname.endswith("_right.bmp"):
                right_path = os.path.join(sub_path, fname)

        if left_path and right_path:      # only proceed if both exist
            left_img  = convert_rgb_to_gray(load_image(left_path))
            right_img = convert_rgb_to_gray(load_image(right_path))
            yield sub, left_img, right_img
        else:
            print(f"Warning: Skipping '{sub}': left or right image missing")


# Loop for each pair of left and right images
for folder, left_img, right_img in iter_stereo_pairs("examples"):
    print(f"Processing pair in folder: {folder}")

    # STEP 1: Extract the Harris Corners as features 

    # Tune Harris corners detection parameters:    
    gauss_k = 5               # 5x5
    neigh_w = 7               # [5x5] or [7x7] --> [7x7] is best
    response_k = 0.12         # between [0.04, 0.15] --> 0.12 is best
    response_threshold = 0.01 # response_threshold --> 0.01
    left_corners, left_R = calculate_harris_corners(left_img, gauss_k, neigh_w, response_k, response_threshold)
    right_corners, right_R = calculate_harris_corners(right_img, gauss_k, neigh_w, response_k, response_threshold)
    plot_corners(left_img, left_corners, right_img, right_corners, save_as=f"{folder}_corners.png")

    # STEP 2: Match feature points using region correlation method
    


    





