import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # To save images as BMP

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

def region_correlation_match(left_img, right_img,
                             left_pts, right_pts,
                             patch_w=7,    # use the same neigh_w you used for Harris
                             row_tol=2,    # search ±2 rows
                             corr_thresh=0.8):
    half = patch_w // 2
    matches = []
    
    for (i1, j1) in left_pts:
        # skip points too close to the border
        if i1-half<0 or i1+half>=left_img.shape[0] or \
           j1-half<0 or j1+half>=left_img.shape[1]:
            continue
        
        P1 = left_img[i1-half:i1+half+1, j1-half:j1+half+1].astype(np.float32)
        μ1 = P1.mean()
        P1z = P1 - μ1
        norm1 = np.sqrt((P1z*P1z).sum())
        
        best_r, best_pt = -1, None
        # only search right corners whose row is within ±row_tol
        for (i2, j2) in right_pts:
            if abs(i2 - i1) > row_tol:
                continue
            if j2-half<0 or j2+half>=right_img.shape[1]:
                continue
            
            P2 = right_img[i2-half:i2+half+1, j2-half:j2+half+1].astype(np.float32)
            μ2 = P2.mean()
            P2z = P2 - μ2
            norm2 = np.sqrt((P2z*P2z).sum())
            
            r = (P1z*P2z).sum() / (norm1*norm2 + 1e-8)
            if r > best_r:
                best_r, best_pt = r, (i2, j2)
        
        # accept only strong correlations
        if best_pt is not None and best_r >= corr_thresh:
            matches.append(((i1, j1), best_pt, best_r))
    
    return matches

def plot_feature_matches(left_img, right_img, matches,
                         line_color='yellow', line_width=0.5):
    """
    Draw the left and right gray images side-by-side,
    with a line joining each matched feature.
    matches is a list of ((i1,j1), (i2,j2), corr) tuples.
    """
    # Ensure grayscale is in [0,255]
    L = left_img.astype(np.uint8)
    R = right_img.astype(np.uint8)

    # Stack them horizontally
    h, w = L.shape
    canvas = np.hstack((L, R))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

    # Draw each match
    for ((i1, j1), (i2, j2), r) in matches:
        # left point is (j1, i1), right point is (j2 + w, i2)
        ax.plot([j1, j2 + w],
                [i1, i2],
                '-', 
                linewidth=line_width,
                color=line_color,
                alpha=0.7)

    plt.show()

def plot_corners(gray_left,  corners_left,
                gray_right, corners_right,save_as_folder,
                color_left='lime',  color_right='red',
                plus_size=80):
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

    # Save the images combined for comparison
    fig.savefig(f"examples/{save_as_folder}/{save_as_folder}_corners.png", bbox_inches='tight', pad_inches=0)

    # Save each image individually with corners in .bmp format
    left_save_path = os.path.join("examples", save_as_folder, f"{save_as_folder}_left_corners.bmp")
    right_save_path = os.path.join("examples", save_as_folder, f"{save_as_folder}_right_corners.bmp")
    tmp_left = left_save_path.replace('.bmp','.png')
    tmp_right = right_save_path.replace('.bmp','.png')

    # Save left image with corners
    left_fig, left_ax = plt.subplots(figsize=(5, 5), dpi=120)
    left_ax.imshow(gray_left, cmap='gray', vmin=0, vmax=255)
    left_ax.scatter(xL, yL, c=color_left, marker='+', s=plus_size, linewidths=1)
    left_ax.axis('off')
    left_fig.savefig(tmp_left, bbox_inches='tight', pad_inches=0)
    plt.close(left_fig)

    # Save right image with corners
    right_fig, right_ax = plt.subplots(figsize=(5, 5), dpi=120)
    right_ax.imshow(gray_right, cmap='gray', vmin=0, vmax=255)
    right_ax.scatter(xR, yR, c=color_right, marker='+', s=plus_size, linewidths=1)
    right_ax.axis('off')
    right_fig.savefig(tmp_right, bbox_inches='tight', pad_inches=0)
    plt.close(right_fig)

    img_left = Image.open(tmp_left)
    img_left.save(left_save_path, format='BMP')

    img_right = Image.open(tmp_right)
    img_right.save(left_save_path, format='BMP')

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
    # Visualize the detected corners and save the images
    plot_corners(left_img, left_corners, right_img, right_corners, save_as_folder=folder)

    # Save the number of features and the (i,j) coordinates one feauture per row in ASCII file
    with open(f"examples/{folder}/{folder}_left_features.txt", "w") as f:
        f.write(f"{len(left_corners)}\n")
        for i, j in left_corners:
            f.write(f"{i} {j}\n")
    with open(f"examples/{folder}/{folder}_right_features.txt", "w") as f:
        f.write(f"{len(right_corners)}\n")
        for i, j in right_corners:
            f.write(f"{i} {j}\n")
    
    # STEP 2: Match feature points using region correlation method
    patch_w = 7
    row_tol = 2
    corr_thresh = 0.8
    matches = region_correlation_match(left_img, right_img,
                                        left_corners, right_corners,
                                        patch_w, row_tol, corr_thresh)
    # Visualize the matched features
    plot_feature_matches(left_img, right_img, matches)

    # STEP 3: Build sparse relative‐depth map
    h, w = left_img.shape
    depth_map = np.zeros((h, w), dtype=np.uint8)

    # extract disparities z' = k/(j1 - j2)
    k = 1
    z_primes = np.array([ (k/(j1 - j2)) 
                        for ((i1,j1),(i2,j2),_) in matches ],
                        dtype=np.float32)
    
    # Save the number of matches in the first row
    # i,j,i,j,r,z -> 1 row per feature
    with open(f"examples/{folder}/{folder}_matched_features.txt", "w") as f:
        f.write(f"{len(matches)}\n")
        for i, ((i1, j1), (i2, j2), r) in enumerate(matches):
            f.write(f"{i1} {j1} {i2} {j2} {r} {z_primes[i]}\n")

    if len(z_primes)==0:
        print("No matches → empty depth map")
    else:
        z_min, z_max = z_primes.min(), z_primes.max()
        # avoid divide‐by‐zero if all disparities equal
        denom = z_max - z_min if z_max>z_min else 1.0

        for idx, ((i1, j1), (i2, j2), _) in enumerate(matches):
            z = z_primes[idx]
            # rescale + round to integer
            depth_map[i1, j1]  = 255.0 - np.floor(245.0 * (z - z_min) / denom + 0.5)
        
        # show the sparse relative‐depth map
        plt.figure(figsize=(6,6), dpi=120)
        plt.imshow(depth_map, cmap='gray', vmin=0, vmax=255)
        plt.title("Sparse Relative Depth Map")
        plt.axis('off')
        # Save the image as png and convert it then to bmp, to avoid using cv2
        plt.savefig(f"examples/{folder}/{folder}_sparse_depth_map.png", bbox_inches='tight', pad_inches=0)
        map_path = os.path.join("examples", folder, f"{folder}_sparse_depth_map.png")
        map = Image.open(map_path)
        map_path = map_path.replace('.png','.bmp')
        map.save(map_path, format='BMP')
        plt.show()




    





