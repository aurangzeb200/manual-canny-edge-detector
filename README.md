# Canny Edge Detection from Scratch

A comprehensive Python implementation of the Canny Edge Detection pipeline. This project demonstrates the fundamental steps of edge extraction by implementing each stage manually, avoiding high-level library shortcuts like `cv2.Canny()`.

## 🛠️ The Pipeline

This implementation follows the five classic steps of edge detection:

1. **Noise Reduction:** Applying a Gaussian filter to smooth the image.
2. **Gradient Calculation:** Computing intensity gradients using horizontal ($G_x$) and vertical ($G_y$) masks.
3. **Non-Maximum Suppression (NMS):** Thinning out the edges by suppressing pixels that are not local maxima in the gradient direction.
4. **Double Thresholding:** Categorizing pixels as Strong, Weak, or Non-edges.
5. **Edge Tracking by Hysteresis:** Finalizing edges by connecting weak pixels to strong pixels using 8-connectivity analysis.

## Mathematical Foundation

### 1. Gradient Magnitude
To find the strength of the edges:

$$\huge M = \sqrt{f_x^2 + f_y^2}$$

### 2. Gradient Direction
To determine the edge orientation for NMS:

$$\huge \theta = \arctan2(f_y, f_x)$$



## Project Structure

```text
canny-edge-detector/
├── src/
│   ├── main.py           # Orchestrates the processing loop
│   ├── filters.py        # Gaussian kernels and convolution logic
│   ├── processing.py     # NMS and Hysteresis implementations
│   └── io_utils.py       # Image handling and visualization plots
├── results/              # Generated edge maps and quantized plots
└── requirements.txt      # Dependencies: NumPy, Pillow, Matplotlib

```

## Getting Started

### Installation

```bash
git clone [https://github.com/aurangzeb200/canny-edge-detector.git](https://github.com/aurangzeb200/canny-edge-detector.git)
cd canny-edge-detector
pip install -r requirements.txt

```

### Usage

Run the detector via the command line:

```bash
python -m main.py --input_folder ./images --output_folder ./results --input_ext png --output_ext png

```

## Visualizing Results

<h2 align="center"><b>Canny Edge Detection Pipeline</b></h2>

<p align="center">
  <table>
    <tr>
      <td align="center"><b>Original CT Scan</b></td>
      <td align="center"><b>Quantized Gradients</b></td>
      <td align="center"><b>Final Edge Map</b></td>
    </tr>
    <tr>
      <td><img src="results/ctscan.tif" width="300"></td>
      <td><img src="results/ctscan_quantized_plot_0.5.png" width="300"></td>
      <td><img src="results/ctscan_edges_0.5_100_50.png" width="300"></td>
    </tr>
  </table>
</p>

<p align="center">
  <i>The visualization above shows the transformation from the raw input (left), through the gradient direction quantization (middle), to the final suppressed and thresholded edges (right).</i>
</p>

## Key Implementations

* **Custom Convolution:** Manual padding and sliding window multiplication.
* **Direction Quantization:** Grouping angles into 0°, 45°, 90°, and 135° for precise pixel comparison.
* **Recursive Edge Tracking:** Using a stack-based `follow` function to ensure edge continuity.
