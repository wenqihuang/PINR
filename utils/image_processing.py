from scipy.ndimage import median_filter, map_coordinates
import numpy as np

def apply_median_filter(b0, size=3):
    return median_filter(b0, size=size)

def rotate_grid(x, y, theta, cx, cy):
    x_rot = (x - cx) * np.cos(np.radians(theta)) - (y - cy) * np.sin(np.radians(theta)) + cx
    y_rot = (x - cx) * np.sin(np.radians(theta)) + (y - cy) * np.cos(np.radians(theta)) + cy
    return x_rot, y_rot

def rotate_images(images, thetas):
    rotated_images_list = []
    
    for theta in thetas:
        nx, ny = images[0].shape[:2]
        cx, cy = (nx - 1) / 2, (ny - 1) / 2  # Define the center of rotation
        x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        x_rot, y_rot = rotate_grid(x, y, theta, cx, cy)
        
        # Flatten the rotated coordinates and convert to float32
        coords = np.vstack([x_rot.ravel(), y_rot.ravel()]).astype(np.float32)
        
        # Apply map_coordinates with properly formatted coordinates to each image
        rotated_images = []
        for img in images:
            if img.ndim == 2:
                rotated_img = map_coordinates(img, coords, order=1, mode='constant').reshape(nx, ny)
            else:
                rotated_img = np.stack([map_coordinates(img[..., i], coords, order=1, mode='constant').reshape(nx, ny) for i in range(img.shape[-1])], axis=-1)
            rotated_images.append(rotated_img)
        
        rotated_images_list.append(rotated_images)
    
    return rotated_images_list

def center_zero_pad(images):
    padded_images = []
    for image in images:
        nx, ny = image.shape[:2]
        pad_size = max(nx, ny)
        pad_x = (pad_size - nx) // 2
        pad_y = (pad_size - ny) // 2
        
        pad_width = [(pad_x, pad_size - nx - pad_x), (pad_y, pad_size - ny - pad_y)]
        if image.ndim == 3:
            pad_width.append((0, 0))  # Only pad spatial dimensions, keep channels unchanged
        
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
        padded_images.append(padded_image)
    
    return padded_images
