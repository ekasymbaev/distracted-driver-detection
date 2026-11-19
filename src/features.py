import cv2 #library for OpenCV
import numpy as np
from skimage.feature import hog
from pathlib import Path
from tqdm import tqdm




def extract_hog(image_path, image_size=(128, 96)):
    """
    Extract HOG features from one image.
    """

    # Load image in grayscale
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Resize image
    image = cv2.resize(image, image_size)

    # Extract HOG features
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.astype(np.float32)




def extract_hog_dataset(df, image_dir, image_size=(128, 96)):
    """Extract HOG features from multiple images."""

    X = []
    image_dir = Path(image_dir)

    # Process each image
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        image_path = image_dir / row.classname / row.img

        # Extract features
        features = extract_hog(image_path, image_size)
        X.append(features)

    # Combine features into a matrix
    return np.stack(X)



