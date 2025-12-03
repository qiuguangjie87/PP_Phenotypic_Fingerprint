import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import imageio.v2 as imageio


image_dir = r"LGASSNet_mask"
output_excel = "Output/Color_GLCM.xlsx"

features = []

for filename in os.listdir(image_dir):
    if not filename.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(image_dir, filename)

    img_rgb = imageio.imread(image_path)
    if img_rgb is None:
        print(f"Unable to read image: {image_path}")
        continue


    hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower_green = np.array([20, 5, 5])
    upper_green = np.array([200, 255, 255])
    mask = cv2.inRange(hsv_img, lower_green, upper_green)

    masked_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
    masked_hsv = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)


    region_pixels_rgb = masked_rgb[mask > 0]
    region_pixels_hsv = masked_hsv[mask > 0]

    if region_pixels_rgb.shape[0] == 0:
        print(f"No valid mask area: {filename}")
        continue


    rgb_mean = np.mean(region_pixels_rgb, axis=0)
    rgb_std = np.std(region_pixels_rgb, axis=0)


    hsv_mean = np.mean(region_pixels_hsv, axis=0)
    hsv_std = np.std(region_pixels_hsv, axis=0)


    gray = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)
    gray = np.uint8(gray)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    features.append({
        "filename": filename,
        "R_mean": rgb_mean[0], "G_mean": rgb_mean[1], "B_mean": rgb_mean[2],
        "R_std": rgb_std[0], "G_std": rgb_std[1], "B_std": rgb_std[2],
        "H_mean": hsv_mean[0], "S_mean": hsv_mean[1], "V_mean": hsv_mean[2],
        "H_std": hsv_std[0], "S_std": hsv_std[1], "V_std": hsv_std[2],
        "GLCM_contrast": contrast,
        "GLCM_homogeneity": homogeneity,
        "GLCM_energy": energy,
        "GLCM_correlation": correlation
    })


cv2.destroyAllWindows()

df = pd.DataFrame(features)
df.to_excel(output_excel, index=False)
print(f"Feature extraction completed and saved to：{output_excel}")

