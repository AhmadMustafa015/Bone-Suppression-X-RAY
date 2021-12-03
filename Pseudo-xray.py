#TODO: 1- Select images only < 1mm slice thickness
#TODO: 2- Use interpolation to convert the voxels into isotropic voxels.
#TODO: 3- Project the image into 2D plane on the antroposterior direction.
#TODO: 4- Non-Linear transformation for the pixels into the XRAY intensity.
#TODO: 5- Bone enhancement; the bones in Xray are clearer than the ones in CT.
#TODO: 6- Unsharp masking process to control the image sharpness in compared to the Xray images.
#TODO: 7- Normalize the image to be between 0 and 1023. The generated image is 10 bit image.
from scipy import ndimage

import settings
import SimpleITK as sitk
from multiprocessing import pool
import glob
import os
import numpy as np
class LoadCtImages:
    """
    Load CT images and select only the images with slice thickness < settings.min_slice_thick for further
    processing
    """
    def __init__(self,input_dir):
        self.input_dir = input_dir
        self.loaded_images = []
        self.interpolation_mode = 'linear'
    def load_images(self):
        slices = []
        scan_extension = ""
        totalNumSlices = 0
        path_to_img = self.input_dir
        for filename in os.listdir(path_to_img):
            if filename.endswith(".dcm"):
                #Reading file in .dcm format
                slices.append(os.path.join(path_to_img, filename))
                scan_extension = "dcm"
                totalNumSlices += 1
                #Add the slice to be precessed in the directory
            elif filename.endswith(".mhd"):
                #Reading file in .mhd format
                slices.append(os.path.join(path_to_img, filename))
                scan_extension = "mhd"
                #Add the raw mhd image to be precessed in the directory
            else:
                print('Unknown input file format; File:\t', os.path.join(path_to_img, filename))
        if len(slices) < 1:
            print('No images found in the directory:\t', path_to_img)

        if scan_extension == "dcm":
            patient_id = path_to_img.basename(path_to_img)
            reader = sitk.ImageSeriesReader()
            itkimage = sitk.ReadImage(reader.GetGDCMSeriesFileNames(path_to_img, patient_id))
        elif scan_extension == "mhd":
            itkimage = sitk.ReadImage(path_to_img)
            patient_id = os.ntpath.basename(path_to_img).replace('.mhd', '')
        spacing_x, spacing_y, spacing_z = itkimage.GetSpacing()
        #Convert the image to numpy array [z,y,x]
        numpyImage = sitk.GetArrayFromImage(itkimage)
        if spacing_z <= settings.min_slice_thick:
            self.loaded_images.append(numpyImage)

    def extract_bones(self,input_image):
        """
        image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
        HU_min: float, min HU value.
        HU_max: float, max HU value.
        HU_nan: float, value for nan in the raw CT image.
        """
        CT_bone = input_image
        # Get bone only
        CT_bone[np.where(np.logical_or(CT_bone<self.HU_min,
                                             CT_bone>self.HU_max))] = self.HU_none
        return CT_bone
    def resample_image(self,input_image,spacing,new_spacing = [1.0, 1.0, 1.0]):
        new_shape = np.round(input_image.shape * spacing / new_spacing)

        # the actual spacing to resample.
        resample_spacing = spacing * input_image.shape / new_shape
        resize_factor = new_shape / input_image.shape
        order_interp = 3
        image_new = ndimage.interpolation.zoom(input_image, resize_factor,
                                               mode=self.interpolation_mode, order=order_interp)
        return image_new
    ####Generation of Pseudo Chest X-ray Images from Computed Tomographic######
    ########Images by Nonlinear Transformation and Bone Enhancement#########

if __name__ == "__main__":
    # Step 1: Select load images and select only images with slice thickness < 1mm
