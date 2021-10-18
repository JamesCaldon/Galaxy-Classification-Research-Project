import pandas as pd
import numpy as np

mesh = 100


def load_dataset(filepath = "..\\data\\raw\\testing\\nair_abraham_2010", count=None, skip=0):
    import os
    def load_files_in_directory(folderpath):
        x = []
        with os.scandir(folderpath) as dirs:
            for entry in dirs:
                if entry.is_file() and entry.name.endswith(".dat"):
                    loaded_file = pd.read_csv(entry.path, header=None, sep='\s+', skiprows=skip*mesh*mesh, nrows=count).to_numpy()
                    x = np.append(x, loaded_file)  
                if entry.is_dir():
                    loaded_file = load_files_in_directory(entry.path)
                    x = np.append(x, loaded_file)  
        return np.array(x)
                    
    x = []; Y = []
    
    
    with os.scandir(filepath) as dirs:

        for entry in dirs:
            if entry.is_dir() and entry.name.__contains__("unknown"):
                x = np.append(x, load_files_in_directory(entry.path))
                
            elif entry.is_dir() and entry.name.__contains__("no_disc"):
                loaded_no_discs = load_files_in_directory(entry.path)
                x = np.append(x, loaded_no_discs)
                Y = np.append(Y, np.full(int(loaded_no_discs.shape[0]/(mesh*mesh)), "E"))
                
            elif entry.is_dir() and entry.name.__contains__("disc"):
                loaded_discs = load_files_in_directory(entry.path)
                x = np.append(x, loaded_discs)
                Y = np.append(Y, np.full(int(loaded_discs.shape[0]/(mesh*mesh)), "ES"))
                
    return np.array(x).reshape(-1, mesh, mesh, 1), np.array(Y)


def raw_to_fits(filepath = "..\\data\\raw\\testing\\nair_abraham_2010", name="nair_abraham_2010.fits"):
    from astropy.io import fits
    from astropy.table import Table
    from matplotlib import pyplot as plt
    image_data, Y = load_dataset(filepath)
    plt.imshow(image_data[0])

    class_data = Table.from_pandas(pd.DataFrame(Y, columns=["class"]))
    primary_hdu = fits.PrimaryHDU(image_data)
    table_hdu = fits.table_to_hdu(class_data)
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(name, overwrite=True)
    
    with fits.open(name) as hdul:
        img_arr = np.array(hdul[0].data)
        img_class = hdul[1].data
    print(img_class)

    from matplotlib import pyplot as plt
    plt.imshow(img_arr[0])


def raw_califa_data_to_fits():
    from astropy.io import fits
    from astropy.table import Table

    ES = pd.read_csv(r"..\data\raw\testing\dr15\disc\ES_SDSS_metadata.txt", sep=',', usecols=[1, 2, 3])
    ES["class"] = "BD/ES"
    E = pd.read_csv(r"..\data\raw\testing\dr15\no_disc\E_SDSS_metadata.txt", sep=',', usecols=[1, 2, 3])
    E["class"] = "E"
    ES_E = ES.append(E)
    fits_BD_E = Table.from_pandas(ES_E)
    print(fits_BD_E)
    DR15, y = load_dataset(filepath = "..\\data\\raw\\testing\\dr15")

    primary_hdu = fits.PrimaryHDU(DR15)
    table_hdu = fits.table_to_hdu(fits_BD_E, character_as_bytes=True)
    print(table_hdu)
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto('califa.fits', overwrite=True)

    CALIFA_FP = 'califa.fits'
    with fits.open(CALIFA_FP) as hdul:
        hdul.info()

        img_arr = np.array(hdul[0].data)
        img_class = hdul[1].data[0]
    print(img_class)
    from matplotlib import pyplot as plt
    plt.imshow(img_arr[0])


def raw_graham_data_to_fits():
    from astropy.io import fits
    from astropy.table import Table

    ES = pd.read_csv(r"..\data\raw\testing\graham\disc\ES_graham_metadata.txt", sep=',', usecols=[1, 2, 3, 4, 5])
    ES.rename(columns = {"Type":"class"}, inplace = True)
    ES.rename(columns = {"Galaxy":"name"}, inplace = True)
    #ES["class"] = "ES"
    E = pd.read_csv(r"..\data\raw\testing\graham\no_disc\E_graham_metadata.txt", sep=',', usecols=[1, 2, 3, 4, 5])
    E.rename(columns = {"Type":"class"}, inplace = True)
    E.rename(columns = {"Galaxy":"name"}, inplace = True)
    #E["class"] = "E"
    ES_E = ES.append(E)
    fits_BD_E = Table.from_pandas(ES_E)
    print(fits_BD_E)
    graham, y = load_dataset(filepath = "..\\data\\raw\\testing\\graham")

    primary_hdu = fits.PrimaryHDU(graham)
    table_hdu = fits.table_to_hdu(fits_BD_E, character_as_bytes=True)
    print(table_hdu)
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto('graham.fits', overwrite=True)

    graham_fp = 'graham.fits'
    with fits.open(graham_fp) as hdul:
        hdul.info()

        img_arr = np.array(hdul[0].data)
        img_class = hdul[1].data[0]
    print(img_class)
    from matplotlib import pyplot as plt
    plt.imshow(img_arr[0])


