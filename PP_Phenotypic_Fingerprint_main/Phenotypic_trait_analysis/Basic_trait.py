
from PIL import Image
import os
import numpy as np
import cv2
import math
import pandas as pd

def find_LW_positions(img_path, class_plex, LW):

    matrix = np.array(Image.open(img_path).convert('L'), 'f')

    if LW == 'L':
        matrix = matrix
    else:
        matrix = matrix.T

    nonzero_indices = np.where(matrix == class_plex)

    if len(nonzero_indices[0]) > 0:
        first_i, first_j = nonzero_indices[0][0], nonzero_indices[1][0]

        last_i, last_j = nonzero_indices[0][-1], nonzero_indices[1][-1]
    else:
        first_i, first_j, last_i, last_j = 0, 0, 0, 0

    return first_i, first_j, last_i, last_j


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def leaf_area(img_path, mesophyll=38, main_vein=75, lateral_vein=113):
    matrix = np.array(Image.open(img_path).convert('L'), 'f')

    LA_plex = np.count_nonzero(matrix)

    MVA_plex = np.sum(matrix == main_vein)
    LVA_plex = np.sum(matrix == lateral_vein)

    return LA_plex, MVA_plex, LVA_plex


def leaf_perimeter_area_rect(image_path):

    image = cv2.imread(image_path, 0)

    _, binary = cv2.threshold(image, 33, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, (0, 0, 0, 0)

    largest_contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)


    x, y, w, h = cv2.boundingRect(largest_contour)

    return perimeter, area, (x, y, w, h)


def calculate_roundness(area, perimeter):
    if perimeter > 0:
        return (4 * math.pi * area) / (perimeter * perimeter)
    return 0


def calculate_basic_trait(img_dir, output_file, ratio):
    print(f"Start Phenotype Analysis...")

    results = []
    count = 0

    image_files = [f for f in os.listdir(img_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    print(f"Found {len(image_files)} image files")

    for filename in image_files:
        img_path = os.path.join(img_dir, filename)

        try:
            LP_pixels, area_pixels, bbox = leaf_perimeter_area_rect(img_path)
            LP = LP_pixels * ratio

            L_first_i, L_first_j, L_last_i, L_last_j = find_LW_positions(img_path, 38, 'L')
            LL_pixels = calculate_distance((L_first_i, L_first_j), (L_last_i, L_last_j))
            LL = LL_pixels * ratio

            W_first_i, W_first_j, W_last_i, W_last_j = find_LW_positions(img_path, 38, 'W')
            LW_pixels = calculate_distance((W_first_i, W_first_j), (W_last_i, W_last_j))
            LW = LW_pixels * ratio

            LA_plex, MVA_plex, LVA_plex = leaf_area(img_path, mesophyll=38, main_vein=75, lateral_vein=113)
            LA = LA_plex * (ratio ** 2)
            MVA = MVA_plex * (ratio ** 2)
            LVA = LVA_plex * (ratio ** 2)

            LWr = LL / LW if LW > 0 else 0
            PLr = LP / LL if LL > 0 else 0
            RN = calculate_roundness(LA, LP)


            result = {
                'File_name': filename,
                'LL_mm': round(LL, 2),
                'LW_mm': round(LW, 2),
                'LP_mm': round(LP, 2),
                'LA_mm2': round(LA, 2),
                'MVA_mm2': round(MVA, 2),
                'LVA_mm2': round(LVA, 2),
                'LWr': round(LWr, 4),
                'PLr': round(PLr, 4),
                'RN': round(RN, 4),
                'LL_pixels': int(LL_pixels),
                'LW_pixels': int(LW_pixels),
                'LA_pixels': int(LA_plex),
                'Bbox_x': bbox[0],
                'Bbox_y': bbox[1],
                'Bbox_w': bbox[2],
                'Bbox_h': bbox[3]
            }

            results.append(result)
            count += 1

            print(f"File {count}/{len(image_files)} : {filename}")

        except Exception as e:
            print(f"Analysis {filename} error: {str(e)}")
            results.append({
                'File_name': filename,
                'Error': str(e)
            })


    if results:
        df = pd.DataFrame(results)

        if not output_file.endswith('.xlsx'):
            output_file = output_file.rsplit('.', 1)[0] + '.xlsx'

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Phenotypic_Traits', index=False)

            if len(df) > 1 and 'Error' not in df.columns:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                summary_stats = df[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max']).round(3)
                summary_stats.to_excel(writer, sheet_name='Summary_Statistics')

        print(f"\nPhenotype analysis complete！")
        print(f"Total {count} image files")
        print(f"Results saved to: {output_file}")

    else:
        print("No files were analysis.")


def main():
    ratio = 13.5 / 100  # 像素比例尺 根据比色卡设置
    img_dir = 'LGASSNet_Prediction'
    output_file = 'Output/Basic_phenotypic_traits.xlsx'

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    if not os.path.exists(img_dir):
        print(f"Error: Image directory '{img_dir}' does not exist！")
        return

    calculate_basic_trait(img_dir, output_file, ratio)


if __name__ == "__main__":
    main()