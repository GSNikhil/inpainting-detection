import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from PIL import Image
import os

# For semantic segmentation
import torch
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

from skimage import io, color, img_as_float
from scipy.ndimage import gaussian_filter

from skimage.morphology import opening, closing, disk
from skimage import io, color, img_as_float, morphology
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi

import pywt
import dtcwt
import csv
import pickle


# read image of any format - png, bmp, jpg and return the rgb image as numpy array
# normalize the image to [0, 1] range
def read_img(image_path):
    img = Image.open(image_path)
    img = np.array(img)
    img = img / 255.0
    return img


def kmeans(img, n_clusters, display=False):
    data = img.reshape((-1, 1))
    H, W = img.shape

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    clusters = kmeans.fit_predict(data)
    clusters = clusters.reshape((H, W))

    if display:
        plt.imshow(clusters, cmap="Spectral")
        plt.title(f"Input Image Segmentation, n_clusters: {n_clusters}")
        plt.axis('off')
        plt.show()

    return clusters


def simple_segmentation(image, n_clusters=5, feature_space='rgb', smoothing_sigma=1,display=False):
    """
    Perform simple image segmentation using K-means clustering.

    Parameters:
    - image: Input image as a NumPy array.
    - n_clusters: The desired number of clusters (segments).
    - feature_space: The color space to use ('rgb' or 'lab').

    Returns:
    - labels: A 2D array of the same height and width as the image, containing the segment labels.
    """
    # Convert image to float representation
    image = img_as_float(image)
    h, w, c = image.shape

    # Apply image smoothing to reduce noise
    smoothed_image = gaussian_filter(image, sigma=(smoothing_sigma, smoothing_sigma, 0))

    # Choose feature space
    if feature_space == 'lab':
        # Convert to Lab color space for better perceptual similarity
        image_feats = color.rgb2lab(smoothed_image)
    else:
        # Use RGB color space
        image_feats = smoothed_image

    # Reshape the image to a 2D array of pixels and color features
    X = image_feats.reshape((-1, c))

    # Perform K-means clustering
    print("Performing K-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Reshape labels back to image dimensions
    labels = labels.reshape((h, w))

    # Apply improved morphological operations to clean up small regions
    selem = morphology.disk(5)
    labels_cleaned = np.zeros_like(labels)
    min_size = 500  # Adjust this value based on image size

    for i in range(n_clusters):
        mask = (labels == i).astype(np.bool_)
        mask_filled = ndi.binary_fill_holes(mask)
        mask_cleaned = morphology.remove_small_objects(mask_filled, min_size=min_size)
        mask_smooth = morphology.closing(mask_cleaned, selem)
        mask_smooth = morphology.opening(mask_smooth, selem)
        labels_cleaned[mask_smooth] = i

    if display:
        plt.imshow(labels_cleaned, cmap='tab20')
        plt.axis('off')
        plt.title('Segmentation Result')
        plt.show()

    return labels_cleaned

# Performs Log Normalization and z-tranforms and clips data [-1, 2] range
def normalize_coeffs(bands):
    reals = []
    imags = []
    for band in range(bands.shape[-1]):
        real_coeffs = bands[:, :, band].real
        imag_coeffs = bands[:, :, band].imag

        # Normalize
        # Log (1+x) transform
        eps = 1e-8
        real_coeffs = np.log10((real_coeffs) + abs(real_coeffs.min()) + eps)
        imag_coeffs = np.log10((imag_coeffs) + abs(imag_coeffs.min()) + eps)

        # Z-Scaling
        real_coeffs = (real_coeffs - real_coeffs.mean()) / (real_coeffs.std() + 1e-8)
        imag_coeffs = (imag_coeffs - imag_coeffs.mean()) / (imag_coeffs.std() + 1e-8)

        reals.append(real_coeffs)
        imags.append(imag_coeffs)

    return np.asarray(reals), np.asarray(imags)

def display_wavelet_coeffs(reals, imags, img_id, display=False, savepath=""):
    # Clip the data to [-1, 2] range
    reals = np.clip(reals, -1, 2)
    imags = np.clip(imags, -1, 2)
    if display:
        plt.figure(figsize=(12, 4))
        for i, data in enumerate(zip(reals, imags)):
            real, imag = data

            plt.subplot(2, 6, i + 1)
            plt.imshow(real, cmap="Spectral")
            plt.title(f"Real, Band{i+1}")
            plt.axis('off')

            plt.subplot(2, 6, i + 7)
            plt.imshow(imag, cmap="Spectral")
            plt.title(f"Imag, Band{i+1}")
            plt.axis('off')

        plt.suptitle(f"Level 1, Real and Imaginary Bands 1-6 {savepath}")

        if savepath != "":
            # directory = os.path.dirname(savepath)
            directory = '/mntdata/avanti/project/ece251c/op_mat'
            plt.savefig(f"{directory}/{img_id}_sam_bands.png")

        plt.show()

def get_patch_mean_variance(band, patch_size):
    H, W = band.shape
    patch_means = []
    for h in range(0, H, patch_size):
        for w in range(0, W, patch_size):
            patch = band[h:h+patch_size, w:w+patch_size]
            patch_means.append(np.mean(patch))

    var = np.var(patch_means)
    return var

def get_band_variance(mask, reals, imags, patch_size):
    real_var = []
    imag_var = []
    for real, imag in zip(reals, imags):
        # Real first
        band = mask * real
        real_var.append(get_patch_mean_variance(band, patch_size))
        
        # Imag
        band = mask * imag
        imag_var.append(get_patch_mean_variance(band, patch_size))

    return np.asarray(real_var), np.asarray(imag_var)

def compute_threshold_dynamic(mask, real_coeffs, imag_coeffs, patch_size):
    var_reals, var_imag = get_band_variance(mask, real_coeffs, imag_coeffs, patch_size)
    var = np.concatenate((var_imag, var_reals))
    var = np.percentile(var, 95)
    return var

# Return the coeffs of the DT-CWT transform
def wavelet_dtcwt(img):
    assert len(img.shape) == 2, "Not a grayscale image"

    # Initialize the DTCWT transform
    transform = dtcwt.Transform2d()

    # Perform the DTCWT on the image with only 1 level
    coeffs = transform.forward(img, nlevels=1)
    return coeffs.highpasses[0]

def wavelet_general(img, wavelet_type='haar'):

    if wavelet_type == 'dtcwt':
        return wavelet_dtcwt(img)

    LL, (LH, HL, HH) = pywt.dwt2(img, wavelet_type)
    print(LH.shape)
    H, W = LH.shape

    coeffs = np.zeros((H, W, 3))
    coeffs[:, :, 0] = LH
    coeffs[:, :, 1] = HL
    coeffs[:, :, 2] = HH

    return coeffs

def noise_zero(img):
    return np.zeros_like(img)

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        print('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        print('The input of stride must be a integer or a int tuple!')


    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

def noise_estimate(im, pch_size=8):
    '''
    Implement of noise level estimation of the following paper:
    Chen G , Zhu F , Heng P A . An Efficient Statistical Method for Image Noise Level Estimation[C]// 2015 IEEE International Conference
    on Computer Vision (ICCV). IEEE Computer Society, 2015.
    Input:
        im: the noise image, H x W x 3 or H x W numpy tensor, range [0,1]
        pch_size: patch_size
    Output:
        noise_level: the estimated noise level
    '''

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    else:
        im = np.expand_dims(im, axis=0)

    # image to patch
    pch = im2patch(im, pch_size, 3)  # C x pch_size x pch_size x num_pch tensor
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))  # d x num_pch matrix
    d = pch.shape[0]

    mu = pch.mean(axis=1, keepdims=True)  # d x 1
    X = pch - mu
    sigma_X = np.matmul(X, X.transpose()) / num_pch
    sig_value, _ = np.linalg.eigh(sigma_X)
    sig_value.sort()

    for ii in range(-1, -d-1, -1):
        tau = np.mean(sig_value[:ii])
        if np.sum(sig_value[:ii]>tau) == np.sum(sig_value[:ii] < tau):
            return np.sqrt(tau)


# normalise the output between [0,1]
def small_median_filter_residue(band, kernel_size):
    median_blur = cv2.medianBlur(band, ksize=kernel_size)
    first_diff = band - median_blur
    second_diff = cv2.medianBlur(first_diff, ksize=kernel_size)
    band = band - second_diff
    band = (band - band.min()) / (band.max() - band.min())
    return band


def wiener_filter(band, kernel_size=3, noise_variance=0.1):
    local_mean = cv2.blur(band, (kernel_size, kernel_size))
    local_variance = cv2.blur((band - local_mean) ** 2, (kernel_size, kernel_size))
    wiener_filtered = local_mean + (local_variance - noise_variance) / (local_variance + 1e-8) * (band - local_mean)
    return np.clip(wiener_filtered, 0, 1)  # Clipping to avoid overflow


from skimage import restoration
from scipy.signal import convolve2d

# normalise the output between [0,1]
def wiener_filter_updated(img):
    psf = np.ones((5, 5)) / 25
    img = convolve2d(img, psf, 'same')
    rng = np.random.default_rng()
    img += 0.1 * img.std() * rng.standard_normal(img.shape)
    deconvolved_img = restoration.wiener(img, psf, 0.1)
    deconvolved_img = (deconvolved_img - deconvolved_img.min()) / (deconvolved_img.max() - deconvolved_img.min())
    return deconvolved_img

# normalise the output between [0,1]
def median_modified_wiener_filter(band, kernel_size=3, noise_variance=0.1):
    # Step 1: Apply median filter
    median_filtered = cv2.medianBlur(band, kernel_size)
    # Step 2: Apply Wiener filter to median-filtered band
    # wiener_filtered = wiener_filter(median_filtered, kernel_size, noise_variance)
    wiener_filtered = wiener_filter_updated(median_filtered)
    wiener_filtered = (wiener_filtered - wiener_filtered.min()) / (wiener_filtered.max() - wiener_filtered.min())
    return wiener_filtered


def get_mask_from_releavant_bands(bands, kernel_size, display=False):
    if len(bands) == 0:
        print("No bands available. Adjust threshold maybe.")
        return 0

    final_mask = np.zeros_like(bands[0])
    for band in bands:
        band = band.astype(np.float32)
        median_filtered = cv2.medianBlur(band, ksize=kernel_size)
        
        # normalise the output between [0,1]
        median_filtered = (median_filtered - median_filtered.min()) / (median_filtered.max() - median_filtered.min())

        smfr_filtered = small_median_filter_residue(band, kernel_size=kernel_size) 
        # wiener_filtered = wiener_filter(band, kernel_size=kernel_size)
        wiener_filtered = wiener_filter_updated(band)
        median_modified_wiener_filtered = median_modified_wiener_filter(band, kernel_size=kernel_size)

        # filters = [median_filtered, smfr_filtered, wiener_filtered, median_modified_wiener_filtered]
        filters = [median_filtered, smfr_filtered, wiener_filtered, median_modified_wiener_filtered]

        #display filtered images
        if display:
            plt.figure(figsize=(12, 4))
            for i, filter_img in enumerate(filters):
                filter_img_clipped = np.clip(filter_img, -1, 2)
                plt.subplot(1, 4, i + 1)
                plt.imshow(filter_img_clipped, cmap="Spectral")
                plt.title(f"Filter {i+1}")
                plt.axis('off')
                plt.colorbar()
            plt.suptitle("Filtered Images")
            plt.show()

        # noise estimate for the band 
        # largest_noise_estimate = noise_estimate(band, 8)

        avg_filter = np.zeros_like(median_filtered)
        for filter_img in filters:
            avg_filter += filter_img
        avg_filter /= len(filters)
        if display:
            avg_filter_clipped = np.clip(avg_filter, -1, 2)
            plt.imshow(avg_filter_clipped, cmap="Spectral")
            plt.title("Average Filter")
            plt.colorbar()
            plt.axis('off')
            plt.show()


        for filter_img in [avg_filter]:
            filtered_clusters = kmeans(filter_img, n_clusters=3, display=True)

            largest_noise_clus = None
            largest_noise_estimate = 1e-32
            # max_noise = -1e32
            for clus in list(np.unique(filtered_clusters)):
            #     internal_mask = (filtered_clusters == clus).astype(np.int8)
            #     noise = (band - filter_img) * internal_mask
            #     noise_range = noise.max() - noise.min()
                noise_est = noise_estimate(band, 8)
                if noise_est > largest_noise_estimate:
                    largest_noise_clus = clus
                    largest_noise_estimate = noise_est

            assert largest_noise_clus != None

            # display the lowest noise cluster
            if display:
                plt.imshow(filtered_clusters == largest_noise_clus)
                plt.title("Largest Noise Cluster")
                plt.colorbar()
                plt.axis('off')
                plt.show()

            final_mask += (filtered_clusters == largest_noise_clus).astype(np.int8)

    return final_mask

def detect_inpainting(file_path, 
                      wavelet_type, 
                      noise_func, 
                      img_id,
                      loaded_detections,
                      patch_size=16, 
                      var_threshold='dynamic', 
                      kernel_size=3, 
                      segmentation_method='sam', 
                      display=False):
    # Read Image
    img = read_img(file_path)
    
    # Display Input Image
    if display:
        plt.imshow(img)
        plt.axis('off')
        plt.title("Input Image")
        plt.show()
        print("Image Shape ", img.shape)

    # Convert to grayscale
    gray_img = img.mean(axis=2)
    H, W = gray_img.shape

    # Display Gray Image
    if display:
        plt.imshow(gray_img, cmap="gray")
        plt.axis('off')
        plt.title("Gray Image")
        plt.show()

    # clusters = semantic_segmentation(img, n_clusters=1, display=True)
    clusters = None
    if segmentation_method == 'kmeans':
        clusters = simple_segmentation(img, n_clusters=4, feature_space='rgb', display=display, smoothing_sigma=1)
    elif segmentation_method == "sam":
        assert loaded_detections != None, "Loaded Detections are not passed"
        masks = loaded_detections["masks"]
        scores = loaded_detections["scores"]
        bounding_boxes = loaded_detections["bounding_boxes"]

        segments = np.zeros((H, W), dtype=int)

        for i, mask in enumerate(masks):
            # Ensure the mask is binary and within bounds
            mask = mask.astype(int)

            # Add the mask to the `segments` array with a unique label
            segments[mask == 1] = i + 1  # Use `i + 1` to avoid overlapping with background (0)

        # Perform segmentation
        clusters = segments
        plt.imshow(clusters, cmap="tab20")
        plt.axis('off')
        plt.title('Segmentation Result')
        plt.show()

    # TODO: Check if this is correct
    # We are W/2 and H/2 because the bands are of this size. But is this correct?
    # assert clusters != None
    clusters = clusters[::2, ::2]

    # Add Noise TODO: Check if this is the right place to add noise
    noise = noise_func(gray_img)
    gray_img += noise

    # DTCWT on grayscale image
    wavelet_coeffs = wavelet_general(gray_img, wavelet_type=wavelet_type) 
    print("Wavelet Coeffs Shape: ", wavelet_coeffs.shape)
    ##########################
    # IMPORTANT: wavelet_coeffs has to of the shape [H/2, W/2, bands]
    ##########################
    # assert wavelet_coeffs.shape == ((H+1)//2, (W+1)//2, 6), f"wavelet_coeffs is of shape: {wavelet_coeffs.shape}, but it's supposed to be {((H+1)//2, (W+1)//2, 6)}"

    # Normalize
    real_coeffs, imag_coeffs = normalize_coeffs(wavelet_coeffs)

    # Display Normalized and Clipped Wavelet Coefficients
    display_wavelet_coeffs(real_coeffs, imag_coeffs, img_id, display=display, savepath=file_path)

    final_mask = np.zeros_like(real_coeffs[0]) # TODO: Put correct shape here

    relavant_band_real = []
    relavant_band_imag = []

    for cluster in list(np.unique(clusters)):
        mask = (clusters == cluster).astype(np.int8)

        #pad the mask to make it equal to the size of the bands
        pad_x = (real_coeffs[0].shape[0] - mask.shape[0])
        pad_y = (real_coeffs[0].shape[1] - mask.shape[1])
        mask = np.pad(mask, ((pad_x//2, pad_x-(pad_x//2)), (pad_y//2, pad_y-(pad_y//2))), 'constant', constant_values=(0, 0))

        #display mask
        if display:
            plt.imshow(mask, cmap="gray")
            plt.axis('off')
            plt.title("Mask Image")
            plt.show()
        
        real_vars, imag_vars = get_band_variance(mask, real_coeffs, imag_coeffs, patch_size=patch_size)

        print("Real Vars: ", real_vars)
        print("Imag Vars: ", imag_vars)

        # if (np.all(real_vars < 1e-3) and np.all(imag_vars < 1e-3)):
        #     print("No relevant bands for this cluster found. Skipping...")
        #     continue

        real_vars_norm = (real_vars - real_vars.min()) / (real_vars.max() - real_vars.min())
        imag_vars_norm = (imag_vars - imag_vars.min()) / (imag_vars.max() - imag_vars.min())

        print("Real Vars Normalized: ", real_vars_norm)
        print("Imag Vars Normalized: ", imag_vars_norm)

        if var_threshold == 'dynamic':
            # var_threshold = compute_threshold_dynamic(mask, real_coeffs, imag_coeffs, patch_size)
            var_threshold_cluster = np.concatenate((real_vars_norm, imag_vars_norm))
            var_threshold_cluster = np.percentile(var_threshold_cluster, 95)

        print("Threshold: ", var_threshold_cluster)

        print("Picked Bands:")
        for i in range(len(real_vars_norm)):
            rvar, ivar = real_vars_norm[i], imag_vars_norm[i]

            if rvar >= var_threshold_cluster:
                relavant_band_real.append((mask * real_coeffs[i], real_vars[i]))
                print(f"Real {i+1}")

            if ivar >= var_threshold_cluster:
                relavant_band_imag.append((mask * imag_coeffs[i], imag_vars[i]))
                print(f"Imag {i+1}")

    # Get the list of variances of the bands
    var_list = [band[1] for band in relavant_band_real] + [band[1] for band in relavant_band_imag]
    var_percentile = np.percentile(var_list, 95)

    print("Releavant Bands", len(relavant_band_real), len(relavant_band_imag))

    # choose the top x percentile bands based on variance 
    relavant_band_real = [band for band in relavant_band_real if band[1] >= var_percentile]
    relavant_band_imag = [band for band in relavant_band_imag if band[1] >= var_percentile]

    #display the relevant bands
    if display:
        for i, band in enumerate(relavant_band_real):
            plt.imshow(band[0], cmap="Spectral")
            plt.title(f"Real Band {i+1}")
            plt.axis('off')
            plt.show()

        for i, band in enumerate(relavant_band_imag):
            plt.imshow(band[0], cmap="Spectral")
            plt.title(f"Imag Band {i+1}")
            plt.axis('off')
            plt.show()

    # Get the bands in a numpy array
    relavant_band_real = [band[0] for band in relavant_band_real]
    relavant_band_imag = [band[0] for band in relavant_band_imag]

    print("Final Releavant Bands", len(relavant_band_real), len(relavant_band_imag))
                
    final_mask += get_mask_from_releavant_bands(relavant_band_real, kernel_size=kernel_size, display=display)
    final_mask += get_mask_from_releavant_bands(relavant_band_imag, kernel_size=kernel_size, display=display)

      
    plt.imshow(final_mask, cmap="gray")
    plt.axis('off')
    plt.title("Final Mask")
    # directory = os.path.dirname(file_path)
    directory = '/mntdata/avanti/project/ece251c/op_mat'
    plt.savefig(f"{directory}/{img_id}_sam_final_mask_pre_morph.png")
    plt.show()

    return final_mask


pkl_folder = "/mntdata/lama_pkl"
img_folder = "/mntdata/avanti/Lama"

img_files = [f for f in os.listdir(img_folder) if f.endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
pkl_files = [f for f in os.listdir(pkl_folder) if f.endswith('.pkl')]

# map unique identifiers to file paths
img_map = {os.path.splitext(img)[0]: os.path.join(img_folder, img) for img in img_files}
pkl_map = {os.path.splitext(pkl.replace('_filtered', ''))[0]: os.path.join(pkl_folder, pkl) for pkl in pkl_files}

for img_id, img_path in img_map.items():
    if img_id in pkl_map:  # match based on the unique identifier
        pkl_path = pkl_map[img_id]
        loaded_detections = None

        with open(pkl_path, "rb") as f:
            loaded_detections = pickle.load(f)
        
        final_mask = detect_inpainting(
            file_path=img_path,
            wavelet_type='dtcwt',
            img_id = img_id,
            noise_func=noise_zero,
            loaded_detections=loaded_detections,
            kernel_size=5,
            patch_size=64,
            var_threshold='dynamic',
            segmentation_method='sam',
            display=True
        )
        print(f"Processed {img_path} and {pkl_path}")
    else:
        print(f"No matching pickle found for {img_path}")


# loaded_detections = None
# with open("../lama.pkl", "rb") as f:
#     loaded_detections = pickle.load(f)
# final_mask = detect_inpainting(file_path="../lama.bmp", wavelet_type='dtcwt', noise_func=noise_zero, loaded_detections=loaded_detections, kernel_size=5, patch_size=64, var_threshold='dynamic', segmentation_method='kmeans', display=True)





































