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
import math
import cv2 as cv
import os
import tempfile
import datetime

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
class CTtoXRAY:
    """
    Load CT images and select only the images with slice thickness < settings.min_slice_thick for further
    processing
    """
    def __init__(self,input_dir=None):
        self.input_dir = input_dir
        self.loaded_images = []
        self.interpolation_mode = 'linear'
        self.HU_min = 205 #200 IN THE PAPER
        self.HU_max = 3000
    def load_images(self,path=None):
        slices = []
        scan_extension = ""
        totalNumSlices = 0
        if path is None:
            path_to_img = self.input_dir
        else:
            path_to_img = path
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
        if path is not None:
            return numpyImage, [spacing_z,spacing_y,spacing_x]

    def extract_bones(self,input_image):
        """
        image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
        HU_min: float, min HU value.
        HU_max: float, max HU value.
        HU_nan: float, value for nan in the raw CT image.
        """
        CT_bone = input_image
        # Get bone only (Binarize the bones)
        CT_bone[np.where(np.logical_or(CT_bone<self.HU_min,
                                             CT_bone>self.HU_max))] = 0
        CT_bone[np.where(np.logical_or(CT_bone>=self.HU_min,
                                       CT_bone<=self.HU_max))] = 1
        # Closing process; erosion and dilation
        kernel = np.ones((5, 5), np.uint8)
        CT_bone_close_binary = cv.morphologyEx(CT_bone, cv.MORPH_CLOSE, kernel)
        CT_bone_close_binary[np.where(CT_bone_close_binary < 0.5)] = 0
        CT_bone_close_binary[np.where(CT_bone_close_binary >= 0.5)] = 1
        # Mask the bone
        assert input_image.shape() == CT_bone_close_binary.shape(), "Error: bone mask size not equal CT scan size"
        ct_bone = input_image * CT_bone_close_binary
        return ct_bone
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
    def projection_2d(self,image):
        """
        :param image: input image [z,y,x]
        :return:
        P (u, w) = sum(CT (u, v, w)) equation 1
        m_avg: is the pixel value at position (u, w) in the integral image of the linear attenuation
        coefficient.
        m_w: linear attenuation coefficient of water (m_w = 0.00195 m−1).
        I_u_v: is the ratio of incident X-ray intensity to the X-ray intensity after substance transmission
        I_trans: is the value of I_u_v after the non_linear transformation using sigmoid function
        """
        Nv = image.shape[1]
        m_w = 0.00195
        a = -4.0 # To control  the slop of the sigmoid curve
        b = 2.0 # To control  the slop of the sigmoid curve
        #WARNING: input image is [z,y,x]
        p_u_w = np.sum(image,axis=1) # sum the pixels along the anteroposterior direction
        #equation (2)
        p_avg = p_u_w / Nv
        #equation (3)
        m_avg = (p_avg * m_w) / 1000.0 + m_w
        m_total = m_avg * Nv
        I_u_v = math.exp(-m_total)
        I_trans = 1/(1+np.exp(-a*(I_u_v-b*np.mean(I_u_v))))
        return I_trans
    def bone_enhancement(self,image,bone,wb=1):
        """
        To make CT bones as clear as in X-ray
        :param image: non-linear transformed image
        bone: non-linear transformed extracted bone
        wb: enhancement coefficient
        :return: bone-enhanced image
        """
        image = image + wb * bone
        return  image
    def unsharp_masking_process(self,image,wu=0.7):
        """
        The unsharp mask process is a method wherein the high frequency
        component obtained by subtracting the low frequency
        component from the original image is multiplied by the
        enhancement coefficient and added to the original image
        :param image: original image
        :param wu: enhancement coefficient
        :return:
        """
        kernel = np.ones((3, 3), np.uint8)
        image_smooth = cv.filter2D(image,-1,kernel)
        unsharp_image = image + wu * (image - image_smooth)
        return unsharp_image
    def save_xray_image(self,image):
        """
        Normalize image between 0 to 1023 and save it as 10bit image
        :param image:
        :return:
        """
        image_min,image_max = min(image),max(image)
        image_normalize = ((pixel-image_min)/(image_max-image_min) * 1023 for pixel in image)
        ##################### SAVE AS DICOM ###################################
        # Create some temporary filenames
        suffix = '.dcm'
        filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name
        filename_big_endian = tempfile.NamedTemporaryFile(suffix=suffix).name

        print("Setting file meta information...")
        # Populate required values for file meta information
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"

        print("Setting dataset values...")
        # Create the FileDataset instance (initially no data elements, but file_meta
        # supplied)
        ds = FileDataset(filename_little_endian, {},
                         file_meta=file_meta, preamble=b"\0" * 128)

        # Add the data elements -- not trying to set all required here. Check DICOM
        # standard
        ds.PatientName = "Test^Firstname"
        ds.PatientID = "123456"

        # Set the transfer syntax
        ds.is_little_endian = True
        ds.is_implicit_VR = True

        # Set creation date/time
        dt = datetime.datetime.now()
        ds.ContentDate = dt.strftime('%Y%m%d')
        timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
        ds.ContentTime = timeStr

        print("Writing test file", filename_little_endian)
        ds.save_as(filename_little_endian)
        print("File saved.")

        # Write as a different transfer syntax XXX shouldn't need this but pydicom
        # 0.9.5 bug not recognizing transfer syntax
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
        ds.is_little_endian = False
        ds.is_implicit_VR = False

        print("Writing test file as Big Endian Explicit VR", filename_big_endian)
        ds.save_as(filename_big_endian)

        # reopen the data just for checking
        for filename in (filename_little_endian, filename_big_endian):
            print('Load file {} ...'.format(filename))
            ds = pydicom.dcmread(filename)
            print(ds)

            # remove the created file
            print('Remove file {} ...'.format(filename))
            os.remove(filename)



if __name__ == "__main__":
    # Step 1: Select load images and select only images with slice thickness < 1mm
    ct_to_xray = CTtoXRAY()
    image, spacing = ct_to_xray.load_images(path='')
    image = ct_to_xray.resample_image(image,spacing)
    bone_raw = ct_to_xray.extract_bones(image)
    image, bone = ct_to_xray.projection_2d(image), ct_to_xray.projection_2d(bone_raw)
    image = ct_to_xray.bone_enhancement(image,bone)
    image = ct_to_xray.unsharp_masking_process(image)
    ct_to_xray.save_xray_image(image)
    print("Successfully finished")


