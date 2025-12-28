from io_utils import *
from filters import *
from processing import *

import os
import argparse
import numpy as np


def process_image(image_path, output_folder, out_ext):
    basename = os.path.splitext(os.path.basename(image_path))[0]
    img = read_gray_image(image_path)

    sigma_values = [0.5, 1.0, 2.0]
    T_for_size = 0.3
    scale_factor = 255
    hysteresis_pairs = [(100, 50), (150, 75)]

    for sigma in sigma_values:
        print(f"  Processing for sigma={sigma}...")
        
        N, sHalf = calculate_filter_size(sigma, T_for_size)
        Gx_int, Gy_int, sf = calculate_gradient(N, sigma, scale_factor=scale_factor)

        fx, fy = apply_masks(img, Gx_int, Gy_int, sf)
        save_u8_image(normalize_to_u8(fx), os.path.join(output_folder, f"{basename}_fx_{sigma}.{out_ext}"))
        save_u8_image(normalize_to_u8(fy), os.path.join(output_folder, f"{basename}_fy_{sigma}.{out_ext}"))

        M_float, M_u8 = compute_magnitude(fx, fy)
        save_u8_image(M_u8, os.path.join(output_folder, f"{basename}_magnitude_{sigma}.{out_ext}"))

        phi_img, phi_deg = compute_gradient_direction(fx, fy)
        q = quantize_gradient_direction(phi_deg)
        save_u8_image((q.astype(np.uint8) * 85), os.path.join(output_folder, f"{basename}_quantized_{sigma}.{out_ext}"))

        plot_path = os.path.join(output_folder, f"{basename}_quantized_plot_{sigma}.{out_ext}")
        mask_thresh = 0.01 * M_float.max() if M_float.max() > 0 else 0.0
        save_quantized_plot_with_colorbar(q, M_float, plot_path, mask_threshold=mask_thresh)

        suppressed_float, suppressed_u8 = non_maxima_suppression(M_float, q)
        save_u8_image(suppressed_u8, os.path.join(output_folder, f"{basename}_suppressed_{sigma}.{out_ext}"))

        for (Th, Tl) in hysteresis_pairs:
            edges = hysteresis_thresholding(suppressed_float, Th, Tl)
            save_u8_image(edges, os.path.join(output_folder, f"{basename}_edges_{sigma}_{Th}_{Tl}.{out_ext}"))

    print(f"Finished processing: {basename}")


def main():
    parser = argparse.ArgumentParser(description="Canny Edge Detector Assignment Runner")
    parser.add_argument("--input_folder", required=True, help="Path to input folder")
    parser.add_argument("--output_folder", required=True, help="Path to save results")
    parser.add_argument("--input_ext", required=True, help="Extension of input images (e.g. png)")
    parser.add_argument("--output_ext", required=True, help="Extension for output images (e.g. png)")
    args = parser.parse_args()

    ensure_folder(args.output_folder)
    img_paths = list_input_images(args.input_folder, args.input_ext)
    
    if not img_paths:
        print(f"No images with extension '.{args.input_ext}' found in {args.input_folder}")
        return

    for img_path in img_paths:
        process_image(img_path, args.output_folder, args.output_ext)

    print("\nAll done. Results saved in", args.output_folder)


if __name__ == "__main__":
    main()