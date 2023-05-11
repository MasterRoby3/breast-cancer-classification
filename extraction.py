import numpy as np
import cv2 
from matplotlib import pyplot as plt
import pandas as pd # for reading and writing tables
import ntpath
from numpy import where
from collections import Counter

# Import utility functions
from util import *


# Define folder from which data can be downloaded
data_folder = "../training_set/"
benign_folder = data_folder + "benign/"
malignant_folder = data_folder + "malignant/"

# Get count of images
benign, malignant, n_ben, n_mal = getCountOfImage(data_folder)

# Output Log
print("Counted", benign, "benign,", malignant, "malignant: \n-", n_ben, "total benign\n-", n_mal, "total malignant")

size_x ,size_y = 128,128
#create empty array of zeros to store image inside it
study_benign     = np.zeros((benign,size_x,size_y))
mask_benign      = np.zeros((benign,size_x,size_y))
study_malignant  = np.zeros((malignant,size_x,size_y))
mask_malignant   = np.zeros((malignant,size_x,size_y))

labels_benign    = np.zeros(benign)
labels_malignant = np.zeros(malignant)

correspondence_index_ben = np.zeros((n_ben,1))
correspondence_index_mal = np.zeros((n_mal,1))

#start load image
                #0        #1     
classes = ['benign', 'malignant']
label = 0
labels = [] #for classification part
images = [] #for classification part
i_ben = 0
i_mal = 0
for cname in  os.listdir(data_folder):
    for filename in sorted (os.listdir(os.path.join(data_folder,cname))):
        imagePath = data_folder + cname + '/' + filename
        image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        if not '_mask' in filename :
            image = cv2.resize(image,(size_x,size_y))
            image = np.array(image)
            images.append(image)
            if 'benign' in filename:
                correspondence_index_ben[num(filename)-1] = i_ben
                study_benign[i_ben]+= np.array(image)
                labels_benign[i_ben] = int(0)
                i_ben +=1                
            if 'malignant' in filename:
                correspondence_index_mal[num(filename)-1] = i_mal
                study_malignant[i_mal]+= np.array(image)
                labels_malignant[i_mal] = int(1)
                i_mal +=1

    for filename in sorted (os.listdir(os.path.join(data_folder,cname))):
        imagePath = data_folder + cname + '/' + filename
        image = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        if '_mask' in filename :
            image = cv2.resize(image,(size_x,size_y))
            image = np.array(image)
            if filename[0] == 'b':
                ind_ = int(correspondence_index_ben[num(filename)-1])
                mask_benign[ind_]+= np.array(image)
            if filename[0] == 'm' :
                ind_ = int(correspondence_index_mal[num(filename)-1])
                mask_malignant[ind_]+= np.array(image)

mask_benign[mask_benign > 0] = 1
mask_malignant[mask_malignant > 0] = 1

print("\nImported:")
print("- benign shape:", np.shape(study_benign), "\tmask:", np.shape(mask_benign))
print("- malignant shape:", np.shape(study_malignant), "\tmask:", np.shape(mask_malignant))

# Collect studies together
studies = (np.concatenate((study_benign, study_malignant), axis = 0))
masks = (np.concatenate((mask_benign, mask_malignant), axis = 0))
labels = (np.concatenate((labels_benign, labels_malignant), axis = 0))

# import useful packages
from radiomics import featureextractor
# special functions for using pyradiomics
from SimpleITK import GetImageFromArray
import radiomics
from radiomics.featureextractor import RadiomicsFeatureExtractor # This module is used for interaction with pyradiomic
import logging
logging.getLogger('radiomics').setLevel(logging.CRITICAL + 1)  # this tool makes a whole TON of log noise

# Plot Setup Code
# Setup the defaults to make the plots look a bit nicer for the notebook
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})

# Define extractor
extractor = featureextractor.RadiomicsFeatureExtractor(binCount = 128, force2D = True)
extractor.enableFeatureClassByName('shape2D')
extractor.settings

results = extractor.execute(GetImageFromArray(studies[0]), GetImageFromArray((masks[0]).astype(np.uint8)), label = 1)

# Extract features from Benign and Malignant studies
for i in range(len(studies)):
  features_currentStudy = extractor.execute(GetImageFromArray(studies[i]), GetImageFromArray((masks[i]).astype(np.uint8)), label = 1)

  # Stack DataFrames
  if i == 0:
    extracted_features = pd.DataFrame([features_currentStudy])
  else:
    extracted_features = pd.concat( [extracted_features, pd.DataFrame([features_currentStudy])], ignore_index=True )

value_feature_names = [c_col for c_col in extracted_features.columns if (c_col.startswith('original') and '_shape_' not in c_col)]
dataset = extracted_features[value_feature_names]

# Saving files for later use
print("\nCreated dataset with shape:", dataset.shape, "with labels:", np.array(labels).shape)
dataset.to_pickle("image_dataset.pkl")
np.save("labels.npy", labels)
print("Saved dataset with filename: \"image_dataset.pkl\"")


# Extraction for external testing
import pathlib

test_path = "../testing_set/"
test_folder = pathlib.Path(test_path)
study_testset = list(test_folder.glob('P???.png'))

size_x, size_y = 128, 128

# Define tst_images array
tst_studies = np.zeros((100,size_x,size_y))
tst_masks = np.zeros((100,size_x,size_y))

for cname in (study_testset):
    head_, cname_erase = ntpath.split(cname)
    cname_erase = os.path.splitext(cname_erase)[0]
    study_id = int(cname_erase[1:])
    tst_imagePath = test_path + '/' + cname_erase + '.png'
    tst_image = cv2.imread(tst_imagePath,cv2.IMREAD_GRAYSCALE)

    tst_image = cv2.resize(tst_image,(size_x,size_y))
    tst_image = np.array(tst_image)
    tst_studies[study_id - 1] = tst_image

    # Load corresponding mask
    msk_imagePath = test_path + cname_erase + '_mask.png'
    msk_image = cv2.imread(msk_imagePath,cv2.IMREAD_GRAYSCALE)
    msk_image = cv2.resize(msk_image,(size_x,size_y))

    msk_image = np.array(msk_image)
    tst_masks[study_id - 1]+= np.array(msk_image)

tst_studies = tst_studies
tst_masks = tst_masks

tst_masks[tst_masks > 0] = 1

# Extract features from Benign and Malignant studies
for i in range(len(tst_studies)):
  features_currentStudy = extractor.execute(GetImageFromArray(tst_studies[i]),
                    GetImageFromArray((tst_masks[i]).astype(np.uint8)), 
                    label = 1)

  # Stack DataFrames
  if i == 0:
    tst_extracted_features = pd.DataFrame([features_currentStudy])
  else:
    tst_extracted_features = pd.concat( [tst_extracted_features, pd.DataFrame([features_currentStudy])], ignore_index=True )


tst_extracted_features = tst_extracted_features[value_feature_names]
np_extst_features = tst_extracted_features.to_numpy()

# Save on file for reuse
tst_extracted_features.to_pickle("test.pkl")