import cv2
import numpy as np
import math
import os
import pandas as pd
from scipy.stats import *
import warnings

warnings.filterwarnings('ignore')



def calculate_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(p1, p2, p3):


    dot_product = (p1[0] - p2[0]) * (p3[0] - p2[0]) + (p1[1] - p2[1]) * (p3[1] - p2[1])
    cross_product = (p1[0] - p2[0]) * (p3[1] - p2[1]) - (p1[1] - p2[1]) * (p3[0] - p2[0])
    angle = math.atan2(cross_product, dot_product)
    angle_degrees = math.degrees(angle) % 360
    return angle_degrees


def find_contour_points(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    leafmarginALL_with_value = []

    if not contours:
        return contours, None, leafmarginALL_with_value

    contour_return = contours[0]
    for contour in contours:
        if contour_return.shape[0] < contour.shape[0]:
            contour_return = contour

    for point in contour_return:
        x, y = point[0]
        leafmarginALL_with_value.append((x, y))

    return contours, contour_return, leafmarginALL_with_value


def find_minxy_maxxy(coords):
    if not coords:
        return (0, 0), (0, 0), (0, 0), (0, 0)

    min_x_coord = min(coords, key=lambda x: x[0])
    max_x_coord = max(coords, key=lambda x: x[0])
    min_y_coord = min(coords, key=lambda y: y[1])
    max_y_coord = max(coords, key=lambda y: y[1])

    return min_x_coord, max_x_coord, min_y_coord, max_y_coord


def calculate_distances_Center(points, target_point):
    distances = []
    for point in points:
        distance = math.sqrt((point[0] - target_point[0]) ** 2 + (point[1] - target_point[1]) ** 2)
        distances.append(distance)
    return distances

def Circular_prediction(Coordinate):
    L1_L2 = calculate_distance(Coordinate[1], Coordinate[2])
    ot1_ot2 = calculate_distance(Coordinate[3], Coordinate[4])
    ot3_ot4 = calculate_distance(Coordinate[5], Coordinate[6])
    ot5_ot6 = calculate_distance(Coordinate[7], Coordinate[8])

    x1, y1 = Coordinate[3]
    x2, y2 = Coordinate[4]
    ot1_ot2 = abs(x1 - x2)

    if ot1_ot2 < ot3_ot4 and ot5_ot6 < ot3_ot4 and L1_L2 / ot1_ot2 <= 1.0:
        return 'Diamond_shape'
    elif 0.9 <= L1_L2 / ot1_ot2 <= 1.2:
        return "Circular"
    elif 0 < L1_L2 / ot1_ot2 < 0.9:
        return "Horizontal_narrow_circular"
    else:
        return 'Other_circular'


def Oval_prediction(Coordinate):
    L1_L2 = calculate_distance(Coordinate[1], Coordinate[2])
    ot1_ot2 = calculate_distance(Coordinate[3], Coordinate[4])
    ot3_ot4 = calculate_distance(Coordinate[5], Coordinate[6])
    ot5_ot6 = calculate_distance(Coordinate[7], Coordinate[8])

    x1, y1 = Coordinate[3]
    x2, y2 = Coordinate[4]
    ot1_ot2 = abs(x1 - x2)

    if ot1_ot2 < ot3_ot4 and ot5_ot6 < ot3_ot4 and L1_L2 / ot1_ot2 <= 1.0:
        return 'Diamond_shape'
    elif 1.2 < L1_L2 / ot1_ot2 < 1.3:
        return "Transverse_oval"
    elif 1.3 < L1_L2 / ot1_ot2 < 1.8:
        return "Broad_oval"
    elif 1.8 <= L1_L2 / ot1_ot2 < 2.4:
        return "Medium_oval"
    elif 2.4 <= L1_L2 / ot1_ot2 < 3.0:
        return "Narrow_oval"
    else:
        return 'Other_oval'


def Needle_Line_Diamond_prediction(Coordinate, Positive_leaf_tip_angle):
    L1_L2 = calculate_distance(Coordinate[1], Coordinate[2])
    ot1_ot2 = calculate_distance(Coordinate[3], Coordinate[4])
    ot3_ot4 = calculate_distance(Coordinate[5], Coordinate[6])
    ot5_ot6 = calculate_distance(Coordinate[7], Coordinate[8])

    x1, y1 = Coordinate[3]
    x2, y2 = Coordinate[4]
    ot1_ot2 = abs(x1 - x2)

    if ot1_ot2 < ot3_ot4 and ot5_ot6 < ot3_ot4 and L1_L2 / ot1_ot2 <= 1.0:
        return 'Diamond_shape'
    elif Positive_leaf_tip_angle > 90:
        return 'Line_shape'
    elif Positive_leaf_tip_angle <= 90:
        return 'Needle_shape'
    else:
        return 'Other_Needle'


def Triangle_prediction(Coordinate):
    L1, ot3, ot1, ot5, L2, ot4, ot2, ot6 = Coordinate[1], Coordinate[5], Coordinate[3], Coordinate[7], Coordinate[2], \
                                           Coordinate[6], Coordinate[4], Coordinate[8]

    if (ot5[0] < ot1[0] < ot3[0] < L1[0] and ot6[0] > ot2[0] > ot4[0] > L1[0] and
            L1[1] < ot3[1] < ot1[1] < ot5[1] and L1[1] < ot4[1] < ot2[1] < ot6[1]):
        return 'Triangle_shape'
    else:
        return 'Other_shape'


def analyze_leaf_shape(image_path):
    contours, contour_return_max, leafmarginALL_with_value = find_contour_points(image_path)
    if not leafmarginALL_with_value:
        return None, None, None, None

    min_x_coord, max_x_coord, min_y_coord, max_y_coord = find_minxy_maxxy(leafmarginALL_with_value)

    OL_center_matching_points = (int((min_y_coord[0] + max_y_coord[0]) / 2),
                                 int((min_y_coord[1] + max_y_coord[1]) / 2))

    O, L1, L2 = OL_center_matching_points, min_y_coord, max_y_coord

    indices1 = leafmarginALL_with_value.index(min_x_coord) if min_x_coord in leafmarginALL_with_value else 0
    indices2 = leafmarginALL_with_value.index(max_x_coord) if max_x_coord in leafmarginALL_with_value else len(
        leafmarginALL_with_value) // 2

    indices3 = indices1 + (indices2 - indices1) // 4 if indices1 < indices2 else 0
    indices4 = indices1 + (indices2 - indices1) * 3 // 4 if indices1 < indices2 else len(leafmarginALL_with_value) // 4
    indices5 = indices2 + (len(leafmarginALL_with_value) - indices2) // 4 if indices2 < len(
        leafmarginALL_with_value) else len(leafmarginALL_with_value) * 3 // 4
    indices6 = indices2 + (len(leafmarginALL_with_value) - indices2) * 3 // 4 if indices2 < len(
        leafmarginALL_with_value) else len(leafmarginALL_with_value) - 1

    ot1, ot2 = min_x_coord, max_x_coord
    ot3 = leafmarginALL_with_value[indices3 % len(leafmarginALL_with_value)]
    ot4 = leafmarginALL_with_value[indices4 % len(leafmarginALL_with_value)]
    ot5 = leafmarginALL_with_value[indices5 % len(leafmarginALL_with_value)]
    ot6 = leafmarginALL_with_value[indices6 % len(leafmarginALL_with_value)]

    Coordinate = [O, L1, L2, ot1, ot2, ot3, ot4, ot5, ot6]

    L1_L2 = calculate_distance(Coordinate[1], Coordinate[2])
    x1, y1 = Coordinate[3]
    x2, y2 = Coordinate[4]
    ot1_ot2 = abs(x1 - x2)

    point1, point3 = leafmarginALL_with_value[50], leafmarginALL_with_value[-50]
    point2 = min_y_coord
    negative_leaf_tip_angle = calculate_angle(point1, point2, point3)
    Positive_leaf_tip_angle = 360 - negative_leaf_tip_angle

    leaf_shape = "Unknown"
    if L1_L2 / ot1_ot2 <= 1.2:
        leaf_shape = Circular_prediction(Coordinate)
    elif 1.2 < L1_L2 / ot1_ot2 < 3.0:
        leaf_shape = Oval_prediction(Coordinate)
    elif L1_L2 / ot1_ot2 >= 3.0:
        leaf_shape = Needle_Line_Diamond_prediction(Coordinate, Positive_leaf_tip_angle)

    triangle_shape = Triangle_prediction(Coordinate)
    if triangle_shape == 'Triangle_shape':
        leaf_shape = triangle_shape

    return L1_L2, ot1_ot2, L1_L2 / ot1_ot2, leaf_shape


def analyze_leaf_tip_shape(image_path):
    contours, contour_return_max, leafmarginALL_with_value = find_contour_points(image_path)
    if not leafmarginALL_with_value:
        return None, None, None, None, None, None

    min_x_coord, max_x_coord, min_y_coord, max_y_coord = find_minxy_maxxy(leafmarginALL_with_value)

    point1, point3 = leafmarginALL_with_value[75], leafmarginALL_with_value[-75]
    point2 = min_y_coord

    negative_leaf_tip_angle = calculate_angle(point1, point2, point3)
    Positive_leaf_tip_angle = 360 - negative_leaf_tip_angle

    leaf_tip = 'other_tip_shape'
    if 0 < Positive_leaf_tip_angle < 120:
        leaf_tip = 'acute'
    elif 120 <= Positive_leaf_tip_angle < 150:
        leaf_tip = 'obtuse'
    elif 150 <= Positive_leaf_tip_angle < 200:
        leaf_tip = 'rounded'
    elif 200 <= Positive_leaf_tip_angle <= 360:
        leaf_tip = 'obcordate'

    return point1, point2, point3, negative_leaf_tip_angle, Positive_leaf_tip_angle, leaf_tip


def get_new_djacent_angle(image_path):
    contours, contour_return_max, leafmarginALL_with_value = find_contour_points(image_path)
    if not leafmarginALL_with_value:
        return []

    min_x_coord, max_x_coord, min_y_coord, max_y_coord = find_minxy_maxxy(leafmarginALL_with_value)

    index_min = leafmarginALL_with_value.index(min_x_coord) if min_x_coord in leafmarginALL_with_value else 0
    index_max = leafmarginALL_with_value.index(max_x_coord) if max_x_coord in leafmarginALL_with_value else len(
        leafmarginALL_with_value) // 2

    new_half_filtered_coordinates = leafmarginALL_with_value[index_max:] + leafmarginALL_with_value[:index_min]

    new_filtered_coordinates = new_half_filtered_coordinates[::10]

    djacent_angle = []
    for i in range(0, len(new_filtered_coordinates) - 3, 3):
        point1, point2, point3 = new_filtered_coordinates[i:i + 3]
        angle = calculate_angle(point1, point2, point3)
        djacent_angle.append(angle)

    new_djacent_angle = [360 - element for element in djacent_angle]
    return new_djacent_angle


def B_count_ratio(arr, arr_total):
    Full_edge = Blunt_ruler = Serrated = Heavy_serrated = 0
    scale = 15

    for num in arr:
        if num <= 180 and num > 180 - scale * 1:
            Full_edge += 1
        elif num <= 180 - scale * 1 and num > 180 - (scale * 2 + 5):
            Serrated += 1
        elif num <= 180 - (scale * 2 + 5) and num > 180 - (scale * 4 - 5):
            Blunt_ruler += 1
        elif num <= 180 - (scale * 4 - 5):
            Heavy_serrated += 1

    total = len(arr_total)
    ratio_Full_edge = Full_edge / total if total > 0 else 0
    ratio_Blunt_ruler = Blunt_ruler / total if total > 0 else 0
    ratio_Serrated = Serrated / total if total > 0 else 0
    ratio_Heavy_serrated = Heavy_serrated / total if total > 0 else 0

    return Full_edge, Blunt_ruler, Serrated, Heavy_serrated, ratio_Full_edge, ratio_Blunt_ruler, ratio_Serrated, ratio_Heavy_serrated, total


def C_count_ratio(arr, arr_total):
    Full_edge = Blunt_ruler = Serrated = Heavy_serrated = 0
    scale = 15

    for num in arr:
        if num >= 180 and num < 180 + scale * 1:
            Full_edge += 1
        elif num >= 180 + scale * 1 and num < 180 + (scale * 2 + 5):
            Serrated += 1
        elif num >= 180 + (scale * 2 + 5) and num < 180 + (scale * 4 - 5):
            Blunt_ruler += 1
        elif num >= 180 + (scale * 4 - 5):
            Heavy_serrated += 1

    total = len(arr_total)
    ratio_Full_edge = Full_edge / total if total > 0 else 0
    ratio_Blunt_ruler = Blunt_ruler / total if total > 0 else 0
    ratio_Serrated = Serrated / total if total > 0 else 0
    ratio_Heavy_serrated = Heavy_serrated / total if total > 0 else 0

    return Full_edge, Blunt_ruler, Serrated, Heavy_serrated, ratio_Full_edge, ratio_Blunt_ruler, ratio_Serrated, ratio_Heavy_serrated, total


def analyze_leaf_margin(image_path):
    new_djacent_angle = get_new_djacent_angle(image_path)
    if not new_djacent_angle:
        return [0] * 18

    C_djacent_angle = [value for value in new_djacent_angle if value >= 180]
    B_djacent_angle = [value for value in new_djacent_angle if value < 180]

    B_Full_edge, B_Blunt_ruler, B_Serrated, B_Heavy_serrated, B_ratio_Full_edge, B_ratio_Blunt_ruler, B_ratio_Serrated, B_ratio_Heavy_serrated, B_total = B_count_ratio(
        B_djacent_angle, new_djacent_angle)
    C_Full_edge, C_Blunt_ruler, C_Serrated, C_Heavy_serrated, C_ratio_Full_edge, C_ratio_Blunt_ruler, C_ratio_Serrated, C_ratio_Heavy_serrated, C_total = C_count_ratio(
        C_djacent_angle, new_djacent_angle)

    Leaf_margin = 'Serrated'
    if B_Heavy_serrated + C_Heavy_serrated >= 4:
        Leaf_margin = 'Heavy_serrated'
    elif B_Blunt_ruler + B_Blunt_ruler > 6:
        Leaf_margin = 'Blunt_ruler'
    elif B_Blunt_ruler + B_Blunt_ruler + B_Heavy_serrated + C_Heavy_serrated > 6:
        Leaf_margin = 'Blunt_ruler'
    elif B_total - (B_Full_edge + C_Full_edge) <= 5:
        Leaf_margin = 'Full_edge'

    return [B_total, B_Full_edge, B_Blunt_ruler, B_Serrated, B_Heavy_serrated, B_ratio_Full_edge,
            B_ratio_Blunt_ruler, B_ratio_Serrated, B_ratio_Heavy_serrated, C_Full_edge, C_Blunt_ruler,
            C_Serrated, C_Heavy_serrated, C_ratio_Full_edge, C_ratio_Blunt_ruler, C_ratio_Serrated,
            C_ratio_Heavy_serrated, Leaf_margin]

def vein_shape_classify(image_path):

    image = cv2.imread(image_path)
    if image is None:
        return 0.0, 0.0, 'non-flabellate'

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 38, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    veins = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    lines = cv2.HoughLines(veins, 1, np.pi / 180, 100)

    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            angles.append(angle)

        mean_angle = np.mean(angles)
        std_angle = np.std(angles)

        vein_shape_return = 'non-flabellate'
        if std_angle < 100:
            if mean_angle > 90 and mean_angle < 100:
                vein_shape_return = 'semi-flabellate'
            elif mean_angle > 0 and mean_angle < 90:
                vein_shape_return = 'flabellate'
            else:
                vein_shape_return = 'non-flabellate'
        else:
            vein_shape_return = 'non-flabellate'
    else:
        mean_angle = 0.0
        std_angle = 0.0
        vein_shape_return = 'non-flabellate'

    return mean_angle, std_angle, vein_shape_return

def main():

    img_dir = 'LGASSNet_Prediction'
    output_excel = 'Output/Leaf_DUS_traits.xlsx'

    output_dir = os.path.dirname(output_excel) if os.path.dirname(output_excel) else '.'
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    for ext in ['.png', '.JPG', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        image_files.extend([f for f in os.listdir(img_dir) if f.lower().endswith(ext)])

    if not image_files:
        print(f"directory {img_dir} not found image file")
        return

    print(f"Found {len(image_files)} image files")
    print("Start extraction of DUS traits...")

    all_results = []

    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(img_dir, filename)
        print(f"Processing the {idx}/{len(image_files)}th image:: {filename}")

        ls_results = analyze_leaf_shape(img_path)
        if ls_results[0] is None:
            print(f"  Skipping this file {filename}")
            continue

        leaf_length, leaf_width, leaf_lw_ratio, leaf_shape = ls_results

        lts_results = analyze_leaf_tip_shape(img_path)
        point1, point2, point3, neg_angle, pos_angle, leaf_tip_shape = lts_results

        lms_results = analyze_leaf_margin(img_path)

        mean_angle, std_angle, vein_shape = vein_shape_classify(img_path)

        result = {
            'File_name': filename,

            'Leaf_Length': leaf_length,
            'Leaf_width': leaf_width,
            'Leaf_L/W': leaf_lw_ratio,
            'Leaf_shape': leaf_shape,

            'point1': str(point1),
            'point2': str(point2),
            'point3': str(point3),
            'Negative_leaf_tip_angle': neg_angle,
            'Positive_leaf_tip_angle': pos_angle,
            'Leaf_tip_shape': leaf_tip_shape,

            'Total': lms_results[0],
            'B_Full_edge': lms_results[1],
            'B_Blunt_ruler': lms_results[2],
            'B_Serrated': lms_results[3],
            'B_Heavy_serrated': lms_results[4],
            'B_ratio_Full_edge': lms_results[5],
            'B_ratio_Blunt_ruler': lms_results[6],
            'B_ratio_Serrated': lms_results[7],
            'B_ratio_Heavy_serrated': lms_results[8],
            'C_Full_edge': lms_results[9],
            'C_Blunt_ruler': lms_results[10],
            'C_Serrated': lms_results[11],
            'C_Heavy_serrated': lms_results[12],
            'C_ratio_Full_edge': lms_results[13],
            'C_ratio_Blunt_ruler': lms_results[14],
            'C_ratio_Serrated': lms_results[15],
            'C_ratio_Heavy_serrated': lms_results[16],
            'Leaf_margin': lms_results[17],

            'mean_angle': mean_angle,
            'std_angle': std_angle,
            'classify_vein_shape': vein_shape
        }

        all_results.append(result)


    if all_results:
        df = pd.DataFrame(all_results)

        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Traits', index=False)

            ls_cols = ['File_name', 'Leaf_Length', 'Leaf_width', 'Leaf_L/W', 'Leaf_shape']
            df[ls_cols].to_excel(writer, sheet_name='LS_Traits', index=False)

            lts_cols = ['File_name', 'point1', 'point2', 'point3', 'Negative_leaf_tip_angle',
                        'Positive_leaf_tip_angle', 'Leaf_tip_shape']
            df[lts_cols].to_excel(writer, sheet_name='LTS_Traits', index=False)

            lms_cols = ['File_name', 'Total', 'B_Full_edge', 'B_Blunt_ruler', 'B_Serrated',
                        'B_Heavy_serrated', 'B_ratio_Full_edge', 'B_ratio_Blunt_ruler',
                        'B_ratio_Serrated', 'B_ratio_Heavy_serrated', 'C_Full_edge',
                        'C_Blunt_ruler', 'C_Serrated', 'C_Heavy_serrated',
                        'C_ratio_Full_edge', 'C_ratio_Blunt_ruler', 'C_ratio_Serrated',
                        'C_ratio_Heavy_serrated', 'Leaf_margin']
            df[lms_cols].to_excel(writer, sheet_name='LMS_Traits', index=False)

            lvs_cols = ['File_name', 'mean_angle', 'std_angle', 'classify_vein_shape']
            df[lvs_cols].to_excel(writer, sheet_name='LVS_Traits', index=False)

        print(f"\n Processing complete！")
        print(f"Successfully processed {len(all_results)}/{len(image_files)}  images")
        print(f"The results have been saved to: {output_excel}")
    else:
        print("No trait data extracted.")


if __name__ == "__main__":
    main()