import os
import numpy as np
import cv2
import pandas as pd
from scipy.fftpack import fft
from skimage.feature import local_binary_pattern


class FeatureExtractor:

    def __init__(self, mtd_image_dir, lbphf_image_dir, output_file='MTD_LBPHF_features.xlsx'):
        self.mtd_image_dir = mtd_image_dir
        self.lbphf_image_dir = lbphf_image_dir
        self.output_file = output_file

        self.N = 3072
        self.M = 5
        self.T_s = int(np.log2(self.N / 2))

        self.radius = 3
        self.n_points = 24

    def preprocess_mtd_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 3, 255, cv2.THRESH_BINARY)
        return binary_image

    def preprocess_lbphf_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image

    def extract_contour(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)

        sampled_points = np.array([contour[i * len(contour) // self.N] for i in range(self.N)])
        return sampled_points

    def calculate_triangle_area(self, p1, p2, p3):
        matrix = np.array([
            [p1[0], p1[1], 1],
            [p2[0], p2[1], 1],
            [p3[0], p3[1], 1]
        ])
        return 0.5 * np.linalg.det(matrix)

    def calculate_centroid(self, p1, p2, p3):
        return (p1 + p2 + p3) / 3

    def calculate_center_distance(self, p, centroid):
        return np.linalg.norm(p - centroid)

    def extract_mtd_features(self, sampled_points):
        if sampled_points is None:
            return None

        mtd_features = []

        for i in range(self.N):
            alpha, beta, gamma = [], [], []
            for k in range(1, self.T_s + 1):
                d = 2 ** (k - 1)
                p1 = sampled_points[(i - d) % self.N][0]
                p2 = sampled_points[i][0]
                p3 = sampled_points[(i + d) % self.N][0]

                area = self.calculate_triangle_area(p1, p2, p3)
                alpha.append(abs(area))
                beta.append(1 if area >= 0 else 0)
                centroid = self.calculate_centroid(p1, p2, p3)
                gamma.append(self.calculate_center_distance(p2, centroid))

            mtd_features.append([alpha, beta, gamma])

        mtd_features = np.array(mtd_features)

        for k in range(self.T_s):
            max_alpha = np.max(np.abs(mtd_features[:, 0, k]))
            if max_alpha > 0:
                mtd_features[:, 0, k] /= max_alpha

            max_gamma = np.max(np.abs(mtd_features[:, 2, k]))
            if max_gamma > 0:
                mtd_features[:, 2, k] /= max_gamma

        mtd_fft = np.zeros((3 * self.T_s, self.M))
        for t in range(self.T_s):
            mtd_fft[3 * t:3 * t + 3, :] = np.vstack([
                np.abs(fft(mtd_features[:, 0, t]))[:self.M],
                np.abs(fft(mtd_features[:, 1, t]))[:self.M],
                np.abs(fft(mtd_features[:, 2, t]))[:self.M]
            ])

        return mtd_fft.flatten()

    def extract_lbphf_features(self, gray_image):
        if gray_image is None:
            return None

        lbp = local_binary_pattern(gray_image, self.n_points, self.radius, method='uniform')

        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.n_points + 3),
                                   range=(0, self.n_points + 2))

        lbp_hist = lbp_hist.astype('float')
        if lbp_hist.sum() > 0:
            lbp_hist /= lbp_hist.sum()

        return lbp_hist

    def get_matching_image_files(self):
        mtd_files = {}
        lbphf_files = {}

        for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tif', '.tiff']:
            for file in os.listdir(self.mtd_image_dir):
                if file.lower().endswith(ext):
                    filename_without_ext = os.path.splitext(file)[0]
                    mtd_files[filename_without_ext] = os.path.join(self.mtd_image_dir, file)

        for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.bmp', '.tif', '.tiff']:
            for file in os.listdir(self.lbphf_image_dir):
                if file.lower().endswith(ext):
                    filename_without_ext = os.path.splitext(file)[0]
                    lbphf_files[filename_without_ext] = os.path.join(self.lbphf_image_dir, file)

        common_filenames = set(mtd_files.keys()) & set(lbphf_files.keys())

        matched_pairs = []
        for filename in common_filenames:
            matched_pairs.append({
                'filename': filename,
                'mtd_path': mtd_files[filename],
                'lbphf_path': lbphf_files[filename]
            })

        return matched_pairs

    def extract_features_from_pair(self, mtd_path, lbphf_path, filename):
        features = {'filename': filename}
        print(f"Extract MTD features: {filename}")
        binary_image = self.preprocess_mtd_image(mtd_path)

        if binary_image is not None:
            sampled_points = self.extract_contour(binary_image)
            mtd_features = self.extract_mtd_features(sampled_points)

            if mtd_features is not None:
                for i, value in enumerate(mtd_features):
                    features[f'MTD_Feature_{i}'] = float(value)
            else:
                print(f"  MTD extraction failed")
                mtd_dimension = self.T_s * 3 * self.M
                for i in range(mtd_dimension):
                    features[f'MTD_Feature_{i}'] = 0.0
        else:
            print(f" MTD image read failed")
            mtd_dimension = self.T_s * 3 * self.M
            for i in range(mtd_dimension):
                features[f'MTD_Feature_{i}'] = 0.0

        print(f"Extract LBP-HF features: {filename}")
        gray_image = self.preprocess_lbphf_image(lbphf_path)
        lbphf_features = self.extract_lbphf_features(gray_image)

        if lbphf_features is not None:
            for i, value in enumerate(lbphf_features):
                features[f'LBPHF_Feature_{i}'] = float(value)
        else:
            print(f"  LBP-HFextraction failed")
            for i in range(self.n_points + 2):
                features[f'LBPHF_Feature_{i}'] = 0.0

        return features

    def process_all_images(self):
        matched_pairs = self.get_matching_image_files()

        if not matched_pairs:
            print(f"directories {self.mtd_image_dir} and {self.lbphf_image_dir} no matching image files")
            print(f"MTD number of files: {len(os.listdir(self.mtd_image_dir))}")
            print(f"LBP-HF number of files: {len(os.listdir(self.lbphf_image_dir))}")
            return False

        print(f"Found {len(matched_pairs)} pairs of matching image files")
        print("Start feature extraction...")

        all_features = []
        success_count = 0

        for idx, pair in enumerate(matched_pairs, 1):
            print(f"\nProcessing the {idx}/{len(matched_pairs)} : {pair['filename']}")

            try:
                features = self.extract_features_from_pair(
                    pair['mtd_path'],
                    pair['lbphf_path'],
                    pair['filename']
                )

                if features:
                    all_features.append(features)
                    success_count += 1
                    print(f"  Successful feature extraction")

            except Exception as e:
                print(f"  Processing failed: {pair['filename']} - {str(e)}")

        if all_features:
            self.save_to_excel(all_features, len(matched_pairs))
            return True
        else:
            print("No features were successfully extracted.")
            return False

    def save_to_excel(self, all_features, total_pairs):
        df = pd.DataFrame(all_features)
        output_dir = os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else '.'
        os.makedirs(output_dir, exist_ok=True)

        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Features', index=False)

            mtd_cols = ['filename'] + [col for col in df.columns if col.startswith('MTD_')]
            if mtd_cols:
                df[mtd_cols].to_excel(writer, sheet_name='MTD_Features', index=False)

            lbphf_cols = ['filename'] + [col for col in df.columns if col.startswith('LBPHF_')]
            if lbphf_cols:
                df[lbphf_cols].to_excel(writer, sheet_name='LBPHF_Features', index=False)

            info_df = pd.DataFrame({
                'Parameters': ['MTD_directory', 'LBP-HF_directory', 'Total_number', 'successfully',
                       'MTD_dimension', 'LBP-HF_dimension', 'Output_file'],
                '值': [self.mtd_image_dir, self.lbphf_image_dir, total_pairs, len(all_features),
                      self.T_s * 3 * self.M, self.n_points + 2, self.output_file]
            })
            info_df.to_excel(writer, sheet_name='Process_information', index=False)

            mtd_files = set([os.path.splitext(f)[0] for f in os.listdir(self.mtd_image_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
            lbphf_files = set([os.path.splitext(f)[0] for f in os.listdir(self.lbphf_image_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])

            match_info = {
                'Statistics': ['MTD_Number', 'LBP-HF_Number', 'matching', 'MTD_unique', 'LBP-HF_unique'],
                'Count': [len(mtd_files), len(lbphf_files), len(mtd_files & lbphf_files),
                       len(mtd_files - lbphf_files), len(lbphf_files - mtd_files)]
            }
            match_df = pd.DataFrame(match_info)
            match_df.to_excel(writer, sheet_name='Matching_information', index=False)

        print(f"\n Feature extraction completed!")
        print(f"   Successfully processed: {len(all_features)}/{total_pairs}")
        print(f"   Output file: {self.output_file}")
        print(f"\n Feature dimension information:")
        print(f"   MTD feature dimension: {self.T_s * 3 * self.M}")
        print(f"   LBP-HF feature dimension: {self.n_points + 2}")
        print(f"   Total number of features: {len(df.columns) - 1}")


def main():

    MTD_IMAGE_DIR = "LGASSNet_Prediction"
    LBPHF_IMAGE_DIR = "LGASSNet_mask"
    OUTPUT_FILE = "Output/MTD_LBPHF_Features.xlsx"

    print("=" * 60)
    print(" MTD & LBP-HF Feature Extraction")
    print("=" * 60)
    print(f"MTD image directory: {MTD_IMAGE_DIR}")
    print(f"LBP-HF image directory: {LBPHF_IMAGE_DIR}")
    print("=" * 60)

    if not os.path.exists(MTD_IMAGE_DIR):
        print(f" MTD image directory does not exist: {MTD_IMAGE_DIR}")
        return

    if not os.path.exists(LBPHF_IMAGE_DIR):
        print(f" LBP-HF image directory does not exist: {LBPHF_IMAGE_DIR}")
        return


    extractor = FeatureExtractor(MTD_IMAGE_DIR, LBPHF_IMAGE_DIR, OUTPUT_FILE)


    success = extractor.process_all_images()

    if success:
        print("\n All processing completed!")
    else:
        print("\n Processing failed!")


if __name__ == "__main__":

    main()