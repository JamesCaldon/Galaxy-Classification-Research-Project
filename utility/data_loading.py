"""

FILE IO Helpers for galaxy classification/regression project
By James Caldon 2021

"""
import numpy as np
import pandas as pd
import os
import os.path
from astropy.io import fits
from astropy.table import Table
import logging
from enum import IntEnum

class Classes(IntEnum):
        E = 0
        ES = 1
        U = -1


_PARENT_PATH = os.path.dirname(__file__) + r"\..\data"
_DATA = {
    "nair_abraham_2010": {
            "filepath": r"fits_testing\nair_abraham_2010.fits"
        },
    "nair_abraham_2010_hdf5": {
            "filepath": r"fits_testing\nair_abraham_2010.hdf5"
        },
    "nair_200_hdf5": {
            "filepath": r"fits_testing\nair_200.hdf5"
        },
    "CG_611":{
        
            "filepath": r"nair_abraham_2010\unknown" #TODO
            
            },
    "IC3328":{
        
            "filepath": r"nair_abraham_2010\unknown"#TODO
            
            },
    "NGC_4342":{
        
            "filepath": r"fits_testing\NGC_4342_ES.fits"
            
            },
    "NGC5845":{
        
            "filepath": r"nair_abraham_2010\unknown"#TODO
            
            },
    "califa": {
            "filepath": r"fits_testing\califa.hdf5"
            },
    "califa_s_0.8": {
            "filepath": r"fits_testing\califa_s_0.8.hdf5"
            },
    "califa_s_1.2": {
            "filepath": r"fits_testing\califa_s_1.2.hdf5"
            },
    "califa_s_1.0": {
            "filepath": r"fits_testing\califa_s_1.hdf5"
            },
    "califa_s_1.0_mesh_200": {
            "filepath": r"fits_testing\califa_s_1.0_mesh_200.hdf5"
            },
    "graham": {
            "filepath": r"fits_testing\graham.fits"
            },
    "fd=0.2-0.8": {
            "filepath": r"fits_training\fd=0.2-0.8.fits"
            },
    "fd=0.3-0.9": {
            "filepath": r"fits_training\fd=0.3-0.9.fits"
            },
    "fd=0.3-0.8": {
            "filepath": r"fits_training\fd=0.3-0.8.fits"
            },
    "fd=0.3-0.7": {
            "filepath": r"fits_training\fd=0.3-0.7.fits"
            },
    "fd=0.3-0.7_hdf5": {
            "filepath": r"fits_training\fd=0.3-0.7.hdf5"
            },
    "fd=0.3-0.9_orig_200_hdf5": {
            "filepath": r"fits_training\fd=0.3-0.9_orig_200.hdf5"
            },
    "fd=0.3-0.6": {
            "filepath": r"fits_training\fd=0.3-0.6.fits"
            },
    "fd=0.3-0.5": {
            "filepath": r"fits_training\fd=0.3-0.5.fits"
            },
    "fd=0.5-0.9": {
            "filepath": r"fits_training\fd=0.5-0.9.fits"
            },
    "fd=0.5-0.7": {
            "filepath": r"fits_training\fd=0.5-0.7.fits"
            },
    "fd=0.7-0.9": {
            "filepath": r"fits_training\fd=0.7-0.9.fits"
            },
    "fd=0.9-0.95": {
            "filepath": r"fits_training\fd=0.9-0.95.fits"
            }
}

def load_data(name="nair_abraham_2010", count=None, skip=0):
    with fits.open(os.path.join(_PARENT_PATH , _DATA[name]["filepath"])) as hdul:  
        x = []; Y = []; metadata = [];
        x = np.array(hdul[0].data)
        print(len(hdul))
        print(hdul.info())
        if (name != "NGC_4342"):
            Y = np.array(hdul[1].data["class"])

            if (name=="graham"):
                mask = pd.Series(Y).str.contains('E\d', na=True)
                Y = np.where(mask == True, 0, 1)
            else:
                Y = np.where(Y == "E", 0, 1)

            metadata = hdul[1].data
        print(np.max(x))
    return x, Y, metadata

def load_data_new(name="fd=0.3-0.7", count=None, skip=0):

    with fits.open(os.path.join(_PARENT_PATH , _DATA[name]["filepath"])) as hdul:
        #print(hdul[1].data)
        x = hdul[1].data[:count]
        Y = np.array(np.where(hdul[2].data["class"][:count] == "E", 0, 1), dtype=np.int8)
        metadata = hdul[2].data[:count]
        print(x.shape)
        print(len(hdul))
        for i in range(3, len(hdul)):
            if (i % 2 == 1):
                x = np.append(x, hdul[i].data[:count], axis=0)
                #print("Loaded x of shape: ", hdul[i].data.shape)
            else:
                Y = np.append(Y, np.array(np.where(hdul[i].data["class"][:count] == "E", 0, 1), dtype=np.int8), axis=0)
                metadata = np.append(metadata, np.array(hdul[i].data[:count], dtype=metadata.dtype), axis=0)
                #print("Loaded Y of shape: ", hdul[i].data["class"].shape)

        #Y = np.where(Y == "E", 0, 1)
        print(Y.shape)
    new_shape = list(x.shape)
    new_shape.append(1)
    return x.reshape(new_shape), Y, metadata

class_mappings = {
        'E': 0,
        'ES': 1,
        'U': -1
}


def load_hdf5_data(name="fd=0.3-0.7_hdf5", count=None, skip=0, class_name='class'):
        import pandas as pd
        import h5py
        import shutil
        import random
        # Load Hdf5 Data
        fp = os.path.join(_PARENT_PATH , _DATA[name]["filepath"])
        #shutil.copyfile(fp, os.path.join(_PARENT_PATH, "temp.hdf5"))
        #f = h5py.File(os.path.join(_PARENT_PATH, "temp.hdf5"), 'r')
        f = h5py.File(os.path.join(_PARENT_PATH , _DATA[name]["filepath"]), 'r')

        metadata = pd.read_hdf(fp, key='metadata')

        all_indices = None
        if (count is not None):
                all_indices = np.array([], dtype=int)
                for cl in list(pd.Categorical(metadata[class_name]).categories):
                        indices = metadata.index[metadata[class_name] == cl].tolist()
                        rand_samp = np.sort(random.sample(indices, count))
                        all_indices = np.append(all_indices, rand_samp)
                all_indices = all_indices.flatten().tolist()
                metadata = metadata.iloc[all_indices]
        x = np.array(f[('image_data')])[all_indices].squeeze()
        print(x.shape)
        Y = metadata[class_name].to_numpy()
        for class_mapping in class_mappings.items():
                Y[Y == class_mapping[0]] = class_mapping[1]
        #print(all_indices)
        return x.squeeze(), np.array(Y).squeeze().astype('float32'), metadata

def load_fits_data(name="califa_new", count=None, skip=0):
        from astropy.io import fits
        from astropy.table import Table
        import numpy as np
        with fits.open(os.path.join(_PARENT_PATH , _DATA[name]["filepath"])) as hdul:
                x = []
                Y = []
                metadata = Table(hdul[1].data).to_pandas()
                x, Y = zip(*[(np.array(hdu.data).reshape(-1, 100, 100), class_mappings[hdu.header['CLASS']]) for hdu in hdul if type(hdu) is fits.CompImageHDU or type(hdu) is fits.ImageHDU])
                x = np.concatenate(x, axis=0)
        return x, Y, metadata

def create_datagen(subset):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.8, 1.2],
        rotation_range=179,
        fill_mode='reflect',
        #shear_range=0.4
        #cval = 0
        #validation_split = 0.2
    )
    datagen.fit(subset)
    return datagen

def create_datagen_test(subset):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[0.8, 1.2],
        rotation_range=45,
        fill_mode='wrap',
    )
    datagen.fit(subset)
    return datagen

def create_datagen_califa(subset):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=[0.8, 1.2],
        rotation_range=45,
        fill_mode='constant',
        cval = 0
        #validation_split = 0.2
    )
    datagen.fit(subset)
    return datagen

def create_datagen_zoom(subset):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        zoom_range=[0.5, 0.7],
        rotation_range=359,
        fill_mode='constant',
        cval = 0
        #validation_split = 0.2
    )
    datagen.fit(subset)
    return datagen