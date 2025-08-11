
'''
+
=======================================================================

 NAME:
      build_cube

 DESCRIPTION:
	builds the full spectrum HSI cube from the registered
VNIR and SWIR cubes collected with the DU Headwall HSI system.
- Reads in two the two separate cubes that have already been registered.
- sorts the wavelengths
- uses a weighted sum approach to estimate the reflectance values in the overlap region
- outputs a uint16 ENVI cube scaled by a scale factor, initially set to 10,000.  Note that
the scale factor is written into the ENVI header file under the tage 'reflectance scale factor'

 USES:

numpy
spectralPy (from the python package spectral)
time

 PARAMETERS:


 KEYWORDS:


 RETURNS:
Outputs the full spectrum HSI cube in ENVI format, saves as uint16,
reflectance, scaled by a scale_factor, originally set at 10,000

 NOTES:


 HISTORY:
2023/12/01 - D. Messinger: created, based on previous code by J. Macalintal

=======================================================================
-
'''

#%% Importing Packages 
# import os
# import cv2 as cv2
# from scipy.ndimage import zoom
# import skimage
# import sys
# import matplotlib.pyplot as plt
# import spectral as spec
import numpy as np
from spectral import *
import spectral.io.envi as envi
import time

###
# starting stuff
###
start_time = time.time()
start_hour = time.gmtime().tm_hour
start_min = time.gmtime().tm_min
start_sec = time.gmtime().tm_sec
print('Starting time [GMT]: ', start_hour,':', start_min,':',start_sec)
print('')
###
# set up the input images
###
infolder = '/Volumes/LaCie/Durham_data/HSI/Symeon/'
#infolder = '/Users/dwmpaci/Desktop/3_PROJECTS/0_Durham/HSI_Data/061223-PGL-bookbinding/'
#infolder = infolder + 'Ducky_and_Fragment/'

#VNIR
#vnir_path_hdr = infolder + '2023_12_06_06_18_30_VNIR/VNIR_data_top.hdr'
vnir_path_dat = infolder + 'Symeon-VNIR-2022_09_07_05_43_38/data'
#SWIR
#swir_path_hdr = infolder + '2023_12_06_06_00_21_SWIR/SWIR_data_bottom_warped.hdr'
swir_path_dat = infolder + 'Symeon-SWIR-2022_09_07_06_03_05/data_SWIR_warped'

###
# and the outputs
###
saveimage = 1 # set to 1 to write out the final image
outfolder = infolder # will write to the same place we opened the originals from
full_outfilehdr = outfolder + 'Symeon_FullSpec.hdr'

###
# open up the two files
###
print ('opening VNIR image file: ', vnir_path_dat)
vnir_image = open_image(vnir_path_dat+'.hdr').load()

vnir_nrows = vnir_image.nrows
vnir_ncols = vnir_image.ncols
vnir_nbands = vnir_image.nbands
print('VNIR IMAGE rows, cols, bands: ', vnir_image.nrows, vnir_image.ncols, vnir_image.nbands)
print('')

# and the SWIR
print ('opening SWIR image file: ', swir_path_dat)
swir_image = open_image(swir_path_dat+'.hdr').load()

swir_nrows = swir_image.nrows
swir_ncols = swir_image.ncols
swir_nbands = swir_image.nbands
print('SWIR IMAGE rows, cols, bands: ', swir_image.nrows, swir_image.ncols, swir_image.nbands)
print('')

###
# BEGINNING of MAIN PROGRAM
###

#%% Determining region of spectral overlap and choosing the wavelength of least wavelength difference
print('---> determining region of spectral overlap...', end = '')
# vnir_wvl = np.array(vnir_image.nbands)
# swir_wvl = np.array(swir_image.nbands)

vnir_wvl = np.copy(vnir_image.bands.centers)
swir_wvl = np.copy(swir_image.bands.centers)

### Getting the overlap region
swir_overlap = swir_wvl[np.where(swir_wvl <= vnir_wvl[-1])]
vnir_overlap = vnir_wvl[np.where(vnir_wvl >= swir_wvl[0])]

# get the indices - will need them later for sorting the cube by ascending wavelengths
swir_overlap_indices = np.argwhere(swir_wvl <= vnir_wvl[-1])
vnir_overlap_indices = np.argwhere(vnir_wvl >= swir_wvl[0])

N_vnir_overlap, M_swir_overlap = len(vnir_overlap), len(swir_overlap)
n_overlap = N_vnir_overlap + M_swir_overlap

# these are the indices in the full combined wvl array that are the overlap
# after concatenating the two wvl arrays together
# add the last vnir index and one to the swir overlap indices for sorting later on
swir_overlap_indices_p1 = vnir_overlap_indices[N_vnir_overlap-1]+swir_overlap_indices+1
full_wvl_overlap_indices = np.concatenate((vnir_overlap_indices,swir_overlap_indices))
# flatten this array
full_wvl_overlap_indices = np.reshape(full_wvl_overlap_indices,-1)
print(' ...done <---')

###
# now to sort the wavelength file to get everything in ascending order
###
print('---> Concatenating and sorting the two images... ')
# last index in vnir cube before the overlap region, total # of bands in vnir - # in overlap
last_vnir_b4_overlap = vnir_image.nbands - N_vnir_overlap
# index of first swir band in full set of unsorted overlap indices, the last vnir + 1
first_swir_overlap_in_full = full_wvl_overlap_indices[N_vnir_overlap-1] + 1
# starting index of the swir cube in the full list of indices, last of the overlap indices + 1
first_swir_after_overlap = full_wvl_overlap_indices[n_overlap-1] +1
# last swir overlap band in swir indices
last_full_overlap_band = first_swir_overlap_in_full + M_swir_overlap

swir_wvl_after_overlap = swir_wvl[first_swir_after_overlap:swir_nbands]
swir_wvl_after_overlap_nbands = np.size(swir_wvl_after_overlap)

#-> combined_nbands is sum of both, need this for the sorting.
combined_nbands = vnir_image.nbands + swir_image.nbands
#-> final_combined_nbands is number of wvl bands in final image after spectral resampling
final_combined_nbands = vnir_image.nbands + swir_wvl_after_overlap_nbands

###
# create the final cube array
###
full_cube = np.ndarray((vnir_image.shape[0],vnir_image.shape[1],final_combined_nbands))

###
# get the final wavelength array: vnir_wvl + swir_wvl[first_swir_after_overlap: swir_bil.nbands
# this is for the header file in the final output cube
###
final_wvl = np.concatenate((vnir_wvl,swir_wvl[first_swir_after_overlap:-1]))
final_nbands = np.size(final_wvl)
final_wvl = np.reshape(final_wvl,final_nbands)

print('---> Sorting the wavelength array... ', end = '')
vnir_swir_wvl = np.array(combined_nbands)
vnir_swir_wvl = np.concatenate((vnir_wvl,swir_wvl))

wvl_sorted_indices = np.argsort(vnir_swir_wvl)
vnir_swir_wvl_sorted = vnir_swir_wvl[wvl_sorted_indices]
print('  ... done <--')

print('--> Building the final cube....')

print('   ---> building cubes 1 & 3 ....')
cube1 = vnir_image[:,:,0:last_vnir_b4_overlap]
cube3 = swir_image[:,:,first_swir_after_overlap:-1]
print('Cube1: ', cube1.shape)
print('Cube3: ', cube3.shape)
print('   ... done <---')

###
# spectrally resample the SWIR cube to the VNIR wavelengths, then do
# a linear weighted average of the VNIR and SWIR values on the new grid for cube 2
###

#-> Get the VNIR overlap cube, and wavelength grid, # of wvl in VNIR overlap: vnir_overlap_indices
#print('size of, vnir_overlap: ', np.size(vnir_overlap), vnir_overlap)
n1 = np.shape(vnir_image)[0]
n2 = np.shape(vnir_image)[1]
n3 = np.size(vnir_overlap_indices)
vnir_overlap_cube = np.reshape(vnir_image[:,:,vnir_overlap_indices],[n1,n2,n3])
print('size of vnir overlap cube: ', np.shape(vnir_overlap_cube))

#-> Get the SWIR overlap cube, and wavelength grid: swir_overlap_indices
n4 = np.size(swir_overlap_indices)
swir_overlap_cube = np.reshape(swir_image[:,:,swir_overlap_indices],[n1,n2,n4])
print('size of swir overlap cube: ', np.shape(swir_overlap_cube))

#-> spectrally resample that to the VNIR wavelength grid
### from spectral
resample = BandResampler(swir_overlap, vnir_overlap)
# # resample the entire image
print('---> Resampling the SWIR cube to the VNIR wavelength grid... ', end = '')
new_SWIR_size = swir_overlap_cube.shape
data = swir_overlap_cube.reshape((-1, new_SWIR_size[2]))
new_SWIR_image = resample.matrix.dot(data.T).T
new_SWIR_image = new_SWIR_image.reshape(n1, n2, n3)
print('... done <---')
print('new SWIR image shape: ', new_SWIR_image.shape)

#-> compute the weights for the weighted average between the two
wgt_start = 0.5/(n3-1)
wgt = np.arange(n3)/(n3)+wgt_start

#-> create the new cube, spectrally sampled on the VNIR wvl spacing, as a
#-> weighted average of the two at each wavlength
print('---> Computing weighted reflectance values in overlap region ...', end = '')
cube2 = np.ndarray((vnir_image.shape[0],vnir_image.shape[1],n3))
for i in range(n3):
    cube2[:,:,i] = vnir_overlap_cube[:,:,i] * (1.0-wgt[i]) + new_SWIR_image[:,:,i]*wgt[i]
print(' ... done <---')
###
# concatenate the three cubes
###
print('   ---> concatenating ...')
full_cube = np.concatenate((cube1,cube2,cube3),axis=2)
#print('full_cube data type: ', full_cube.dtype)
#print('Shape of full cube: ', full_cube.shape)
print('   ... done <---')

### try to free up some memory
print('---> freeing up some memory... ', end = '')
del swir_image
del swir_overlap_cube
del vnir_overlap_cube
del new_SWIR_image
del data
del cube1
del cube2
del cube3
print(' ... done <---')

###
# multiply by 10,000 and convert to int
###
print('---> Converting the cube to uint16 * a scale factor for writing....')
scale_factor = 10000
print('      -> scale factor = ',scale_factor)
int_cube = full_cube * scale_factor
int_cube = int_cube.astype('uint16')
#print('int_cube data type: ', int_cube.dtype)
#print('Shape of int cube: ', int_cube.shape)

# free up more memory
del full_cube

###
# save the full concatenated, sorted, cube
###
if saveimage == 1 :
    print('Saving the registered, sorted, full spectrum cube...')
    md = vnir_image.metadata.copy()
    md['wavelength'] = final_wvl
    # md['nrows'] = vnir_img.shape[0]
    # md['ncols'] = vnir_img.shape[1]
    md['dtype'] = 'uint16'
    md['bands'] = final_nbands
    md['reflectance scale factor'] = scale_factor
    print('---> Writing cube to: ', full_outfilehdr, end='')
    envi.save_image(full_outfilehdr, int_cube, force='True', metadata=md)
#    envi.save_image(full_outfilehdr, full_cube, force='True', metadata=md)
    print('  ... done <---')

print("--- %5.2f seconds ---" % (time.time() - start_time))
print('CODE COMPLETION!')
