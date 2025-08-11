'''
+
=======================================================================

 NAME:
      coregister_controlpoints_gui

 DESCRIPTION:
registers a SWIR HSI to the VNIR HSI by allowing the user to choose control points in a GUI window
- reads in the VNIR and SWIR cubes
- displays a GUI with both images side by side
- user chooses control points: note that it is very useful to zoom in to choose
 accurate points; you can zoom with the left mouse button (after selecting the
 magnifying glass in the menu bar) and choose the points in each window with the right mouse button
- computes the homography at the two images nearest to 950 nm
- warps the entire SWIR image with that homography
- outputs a warped SWIR image

 USES:
cv2 (from the package OpenCV)
numpy
matplotlib
sys
spectral (from the python package spectral)
os
time

 PARAMETERS:
needs the paths to the VNIR and SWIR envi header files all the way at the bottom of the code

 KEYWORDS:


 RETURNS:
saves a warped SWIR image to the same directory where the original file is with a
new file name <orig_file_warped.hdr>

 NOTES:
the code assumes that the SWIR image is flipped relative to the VNIR image which should
always be the case unless the camera orientations are changed

 HISTORY:
2023/12/4: created by D. Messinger based largely on code by Amir Hassanzadeh
2023/12/14: D. Messinger: added code to resample the SWIR onto the VNIR spatial grid;
    code taken from J. Macalintal's registration code
2025/08/11: Fei Zhang:

=======================================================================
-
'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
from spectral.io import envi
import os
import time
from scipy.ndimage import zoom
# import rasterio

# def visualize_matches(image1, keypoints1, image2, keypoints2, matches):
#     # Draw matches on a new image
#     matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     plt.imshow(matched_image)
#     plt.show()


# def load_images(vnir_path, swir_path):
#     with rasterio.open(vnir_path) as src:
#         vnir_arr = src.read()
#         vnir_profile = src.profile
#         vnir_wavelengths = np.array([float(i.split(" ")[0]) for i in src.descriptions])
#     with rasterio.open(swir_path) as src:
#         swir_arr = src.read()
#         swir_profile = src.profile
#         swir_wavelengths = np.array([float(i.split(" ")[0]) for i in src.descriptions])
#
#     return (vnir_arr, vnir_profile, vnir_wavelengths), (swir_arr, swir_profile, swir_wavelengths)
#


def load_images_envi(vnir_path, swir_path):

    print('---> reading in images:')
    print('VNIR: ', vnir_path)
    print('SWIR: ', swir_path)

    vnir_ds = envi.open(vnir_path)
    vnir_profile = vnir_ds.metadata
    vnir_wavelengths = vnir_profile["wavelength"]
    vnir_wavelengths = np.array([float(i) for i in vnir_wavelengths])
    vnir_arr = np.transpose(vnir_ds.load(), [2,0,1])

    swir_ds = envi.open(swir_path)
    swir_profile = swir_ds.metadata
    swir_wavelengths = swir_profile["wavelength"]
    swir_wavelengths = np.array([float(i) for i in swir_wavelengths])
    swir_arr = np.transpose(swir_ds.load(), [2,0,1])

    print(' ... done <---')
    print('Shape of VNIR: ', vnir_arr.shape[0], vnir_arr.shape[1], vnir_arr.shape[2])
    print('Shape of SWIR: ', swir_arr.shape[0], swir_arr.shape[1], swir_arr.shape[2])

    return (vnir_arr, vnir_profile, vnir_wavelengths), (swir_arr, swir_profile, swir_wavelengths)

def init_figs(vnir_arr,
              vnir_wavelengths,
              swir_arr,
              swir_wavelengths):

    # picking a band close to
    vnir_pair_index = np.argmin(abs((vnir_wavelengths - 950)))
    swir_pair_index = np.argmin(abs((swir_wavelengths - 950)))

    # picking the at 950, and get an average of window size before and after to reduce noise
    window_size = 25;
    res_vnir = 1.6
    window_size_vnir = int((window_size / res_vnir) / 2)
    res_swir = 6
    window_size_swir = int((window_size / res_swir) / 2)
    vnir_image = np.mean(vnir_arr[vnir_pair_index - window_size_vnir:vnir_pair_index + window_size_vnir],
                         0)  # spectral res: 1.6 nm
    swir_image = np.mean(swir_arr[swir_pair_index - window_size_swir:swir_pair_index + window_size_swir],
                         0)  # spectral res: 6 nm

    # Create a figure with two subplots in a single row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # convert that to uint8 for cv2
    plt.suptitle("Use the right mouse button to pick points; at least 4. \n"
                 "Close the figure when finished.")
    to_uint8 = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    vnir_image_uint8 = to_uint8(vnir_image)
    swir_image_uint8 = np.fliplr(to_uint8(swir_image))

    # Display the VNIR image on the left subplot
    ax1.imshow(vnir_image_uint8)
    ax1.set_title('VNIR Image')

    # Display the SWIR image on the right subplot
    ax2.imshow(swir_image_uint8)
    ax2.set_title('SWIR Image')

    return fig, ax1, ax2, vnir_image, vnir_image_uint8, swir_image, swir_image_uint8


# def save_image(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M):
#     swir_registered_bands = []
#     for i in range(len(swir_wavelengths)):
#         swir_registered_bands.append(
#             cv2.warpPerspective(np.fliplr(swir_arr[i]), M, (vnir_arr.shape[2], vnir_arr.shape[1])))
#
#     # save data
#     import os
#     # output_path = swir_path.replace(".tif", "_warped.tif")
#     output_path = os.path.basename(swir_path.replace(".tif", "_warped.tif"))
#     vnir_profile.update(count=len(swir_registered_bands))
#     with rasterio.open(output_path, 'w', **vnir_profile) as dst:
#         for i, band in enumerate(swir_registered_bands):
#             dst.write_band(i + 1, band)
#
#     from gdal_set_band_description import set_band_descriptions
#     bands = [int(i) for i in range(1, len(swir_wavelengths) + 1)]
#     names = swir_wavelengths.astype(str)
#     band_desciptions = zip(bands, names)
#     set_band_descriptions(output_path, band_desciptions)
#
#
#     print("Registered Image Saved to " + output_path)
#     sys.exit()

def save_image_envi(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M):

# need to upsample the SWIR image to the VNIR image spatial grid first
# Upsample each band
#     print("     ---> Spatially Upsampling Each Band...", end="")
#     print('')
#     print('swir_path:', swir_path)
#     print('bands : samples : lines')
#     print('Shape of VNIR: ', vnir_arr.shape[0], vnir_arr.shape[1], vnir_arr.shape[2])
#     print('Shape of SWIR: ', swir_arr.shape[0], swir_arr.shape[1], swir_arr.shape[2])
#     upsample_ratio = vnir_arr.shape[2] / swir_arr.shape[2]
#     print('upsample ratio: ', upsample_ratio)
#     swir_cube_up = zoom(swir_arr, zoom=[1,upsample_ratio,upsample_ratio])
#     print('  ... done <---')
# # Get the dimensions of the data cube
#     b,c,r = swir_cube_up.shape
#     print('VNIR shape: ', vnir_arr.shape[0], vnir_arr.shape[1], vnir_arr.shape[2])
#     print('Shape of upsampled SWIR cube [r,c,b]: ',r,c,b )
# #    sys.exit()

    swir_registered_bands = []
    for i in range(len(swir_wavelengths)):
        swir_registered_bands.append(
            cv2.warpPerspective(np.fliplr(swir_arr[i]), M, (vnir_arr.shape[2], vnir_arr.shape[1])))
            #cv2.warpPerspective(np.fliplr(swir_cube_up[i]), M, (vnir_arr.shape[2], vnir_arr.shape[1])))
    par_dir = os.path.dirname(swir_path)

    print('Saving the registered SWIR cube...')
    output_path = swir_path.replace(".hdr","_warped.hdr")
    print(output_path)

    # replicating vnir metadata except the bands and wavelength
    metadata = {}
    for k, v in vnir_profile.items():
        if (k != "bands") or (k != "wavelength"):
            metadata[k] = vnir_profile[k]
    metadata["bands"] = str(len(swir_wavelengths))
    metadata["wavelength"] = [str(i) for i in swir_wavelengths]

    swir_arr_out = np.transpose(swir_registered_bands, [1,2,0])
    # envi.save_image(swir_path.replace(".hdr", "_warped_2.hdr"), swir_arr, metadata=metadata, force=True)
    envi.save_image(output_path, swir_arr_out, metadata=metadata, force=True)

def main(vnir_path,swir_path):
    global not_satisfied

    # load images envi
    (vnir_arr, vnir_profile, vnir_wavelengths),\
        (swir_arr, swir_profile, swir_wavelengths) = load_images_envi(vnir_path, swir_path)


    not_satisfied = True
    while not_satisfied:
        fig, ax1, ax2, vnir_image, vnir_image_uint8, swir_image, swir_image_uint8 = init_figs(vnir_arr,
                  vnir_wavelengths,
                  swir_arr,
                  swir_wavelengths)

        vnir_points = []
        swir_points = []

        def on_click_vnir(event):
            if event.inaxes == ax1 and event.button == 3:  # Left mouse button clicked in VNIR subplot
                vnir_points.append((event.xdata, event.ydata))
                ax1.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax1.figure.canvas.draw_idle()

        def on_click_swir(event):
            if event.inaxes == ax2 and event.button == 3:  # Left mouse button clicked in SWIR subplot
                swir_points.append((event.xdata, event.ydata))
                ax2.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax2.figure.canvas.draw_idle()

        # Connect the mouse click events to the respective axes
        fig.canvas.mpl_connect('button_press_event', on_click_vnir)
        fig.canvas.mpl_connect('button_press_event', on_click_swir)
        plt.show()


        print(f"Found points are:\n VNIR: {vnir_points}\n SWIR: {swir_points}")
        
        # Validate that we have the same number of points in both images
        if len(vnir_points) != len(swir_points):
            print(f"Error: Number of VNIR points ({len(vnir_points)}) does not match number of SWIR points ({len(swir_points)})")
            print("Please select the same number of corresponding points in both images.")
            continue
        
        # Check if we have at least 4 points (minimum required for homography)
        if len(vnir_points) < 4:
            print(f"Error: Need at least 4 point pairs for homography calculation. Currently have {len(vnir_points)} pairs.")
            print("Please select at least 4 corresponding points in both images.")
            continue
        
        # calculate homography based on points found
        # point passed to homography should be x, y order
        vnir_points = np.array(vnir_points)
        swir_points = np.array(swir_points)
        M, mask = cv2.findHomography(swir_points, vnir_points, cv2.RANSAC, 5)


        # show the result and see if the use is satisfied
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        def on_key(event):
            global not_satisfied
            if event.key == 'escape':  # Close figure if Escape key is pressed
                not_satisfied = False;
                plt.close(fig)
        ax.imshow(cv2.warpPerspective(np.fliplr(swir_arr[0]), M, (vnir_image.shape[1], vnir_image.shape[0])),
                   alpha=0.5)
        ax.imshow(vnir_image, alpha=0.5)
        ax.set_title('Overlay of Coregistered Image \n'
                     'if satisfied press Escape to save image\n'
                     'if NOT satisfied close the figure to restart.')
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    # save image at last
    save_image_envi(swir_arr, swir_wavelengths, swir_path, vnir_arr, vnir_profile, M)

if __name__ == "__main__":

    start_time = time.time()
    start_hour = time.gmtime().tm_hour
    start_min = time.gmtime().tm_min
    start_sec = time.gmtime().tm_sec
    print('Starting time [GMT]: ', start_hour, ':', start_min, ':', start_sec)

    # vnir_path = "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_10_56_30_VNIR/data.tif"
    # swir_path = "/Volumes/T7/axhcis/Projects/NURI/data/uk_lab_data/cal_test/2023_10_12_11_15_28_SWIR/data.tif"
    # vnir_path = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/HSI_Data/061223-PGL-bookbinding/'
    # vnir_path = vnir_path+ '2023_12_06_06_18_30_VNIR/VNIR_data_bottom.hdr'
    # swir_path = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/HSI_Data/061223-PGL-bookbinding/'
    # swir_path = swir_path + '2023_12_06_06_00_21_SWIR/SWIR_data_bottom.hdr'
    # vnir_path = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/HSI_Data/Ducky_and_Fragment/'
    # vnir_path = vnir_path+ 'VNIR/data_VNIR_cropped.hdr'
    # swir_path = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/HSI_Data/Ducky_and_Fragment/'
    # swir_path = swir_path + 'SWIR/data_SWIR_cropped.hdr'
    vnir_path = '/home/fzhcis/mylab/gdrive/projects_with_Dave/for_Fei/Data/Ducky_and_Fragment/'
    vnir_path = vnir_path+ 'VNIR/data_VNIR_cropped.hdr'
    swir_path = '/home/fzhcis/mylab/gdrive/projects_with_Dave/for_Fei/Data/Ducky_and_Fragment/'
    swir_path = swir_path + 'SWIR/data_SWIR_cropped.hdr'
    main(vnir_path,swir_path)

