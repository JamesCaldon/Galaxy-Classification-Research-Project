from astropy.io.fits import file
from astropy.io.fits.convenience import update
from joblib.parallel import Parallel
from pandas.core.frame import DataFrame


def galaxy_dataframe_to_fits(galaxies, mesh, filepath, name):
    def add_dataframe_to_header(row, hdul):
        header = fits.CompImageHDU().header
        header.extend(fits.Header(row[galaxies.columns.difference(['image'])].to_dict()), strip=False, update=True)
        hdul.append(fits.CompImageHDU(data=row['image'], header=header, name=row['name']))

    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    fits_path = os.path.abspath(os.path.join(filepath, name))
    print("Getting metadata")
    print(type(galaxies))
    metadata = galaxies[galaxies.columns.difference(['image'])]

    print("Converting metadata to Astropy Table")
    bin_hdu = fits.table_to_hdu(Table.from_pandas(metadata))
    primary_hdu = fits.PrimaryHDU(header=fits.Header([fits.Card('mesh', mesh)]))

    hdul = fits.HDUList([primary_hdu, bin_hdu])

    print("Adding dataframes to hdu")
    galaxies.apply(add_dataframe_to_header, axis=1, hdul=hdul)



    print("Writing to file")
    hdul.writeto(fits_path, overwrite=True)

    with fits.open(fits_path) as hdul:
        print("Primary Header \n", repr(hdul[0].header))
        print("First BinTable Data as a pandas dataframe \n", repr(Table(hdul[1].data).to_pandas()))
        print("First Comp Image Hdu Header \n", repr(hdul[2].header))
        plt.imshow(hdul[2].data)

def import_graham_data(mesh=100, save_dir="..\\data\\raw\\testing\\graham", name="graham.fits"):
    print("Starting Graham Import")
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table
    import matplotlib.pyplot as plt
    import pandas as pd
    graham_sample = r"D:\Galaxy-Classification-Research-Project\data\raw\testing\graham\GalaxySample.dat"
    df = pd.read_csv(graham_sample, sep='\s+')
    df['class'] = 'U'
    df['image'] = df.apply(lambda row: np.array(download_SDSS_imgcutout(row["ra"], row["dec"], mesh)), axis = 1)
    galaxy_dataframe_to_fits(df, mesh, save_dir, name)

def import_nair_data(mesh=100, save_dir="..\\data\\fits_testing", name="nair_abraham_2010.hdf5"):
    print("Starting Graham Import")
    import numpy as np
    import pandas as pd
    import io, time
    from PIL import Image
    from PIL import ImageOps
    from concurrent.futures import as_completed
    from pprint import pprint
    from matplotlib import pyplot as plt
    from requests_futures.sessions import FuturesSession
    
    def get_future_from_row(row):
        #print(row)
        future = session.get(f"http://skyserver.sdss.org/dr15/SkyServerWS/ImgCutout/getjpeg?ra={row.RA}&dec={row.DEC}&scale=1.0&height={mesh}&width={mesh}")
        future.row_index = row.Index
        return future


    nair_sample = r"D:\Galaxy-Classification-Research-Project\data\raw\nair_abraham_2010\NairAbrahamMorphology.cat"
    df = pd.read_csv(nair_sample, sep='\s+')
    print(df.head())
    df['gclass'] = 'U'
    df['image'] = None
    df = df[df['TType'] == -5].copy()
    #df['image'] = df.apply(lambda row: np.loadtxt("img\\" + str(row.name) + ".txt"), axis = 1)
    session = FuturesSession()
    #url = r"http://skyserver.sdss.org/dr15/SkyServerWS/ImgCutout/getjpeg?ra=" + str(ra) + r"&dec=" + str(dec) + r"&scale=0.8&height=" + str(mesh) + "&width=" + str(mesh)
    #futures = [session.get(f"http://skyserver.sdss.org/dr15/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale=1.0&height={mesh}&width={mesh}")]
    futures = []
    for row in df.itertuples():
        futures.append(get_future_from_row(row))
        

    for future in as_completed(futures):
        res = future.result()
        img = Image.open(io.BytesIO(res.content))
        img = ImageOps.grayscale(img)
        img = img.resize((mesh, mesh))
        img = np.array(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        print(f"\rFinished Processing Image at url: {res.request.url}", end='       ', flush=True)
        df.at[future.row_index, 'image'] = img
        #time.sleep(10000)
        
    print(df['image'].iloc[0].shape)
    plt.imshow(df['image'].iloc[0])
    df.rename({'#JID': 'JID', 'M/L': 'M_over_L', 'V/Vmax': 'V_over_Vmax'}, inplace=True, axis=1)
    
    galaxy_dataframe_to_hdf5(df, mesh, save_dir, name)

def import_califa_data(mesh=100, save_dir="..\\data\\fits_testing", name="califa.hdf5"):
    print("Starting Califa Import")
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table
    import matplotlib.pyplot as plt
    fits_data = r"E:\OneDrive - The University of Western Australia\Documents\Honours - Galaxy Classification\Galaxy-Classification-Research-Project\data\raw\photometric_decomposition_HT.fits"

    with fits.open(fits_data) as hdul:
        hdu = hdul[0]
        hdu.header

        d = fits.getdata(fits_data, 1)
        t = Table(d)

        df = t.to_pandas()
        df.rename(columns = {'col2_1': 'name', 'col3_1':'ra', 'col4_1':'dec'}, inplace=True)

        df = df[(df["col5_1"] == "U") | ((df["col5_1"] == "K") & (df["col23"] == 1.0))].copy().reset_index(drop=True)

        df['class'] = 'U'
        df['class'] = df['class'].where(((df["col5_1"] != "K") & (df["col23"] != 1.0)), 'E').where(df["col5_1"] != 'U', 'ES')

        df = df[['name', 'ra', 'dec', 'class']].copy()

        print(df)
        
        #df['image'] = df.apply(lambda row: np.array(download_SDSS_imgcutout(row["ra"], row["dec"], mesh)), axis = 1)
        df['image'] = df.apply(lambda row: np.array(download_SDSS_imgcutout(row["ra"], row["dec"], mesh)), axis = 1)
        print(df['image'].iloc[0].shape)
        plt.imshow(df['image'].iloc[0])

        galaxy_dataframe_to_hdf5(df, mesh, save_dir, name)


def download_SDSS_imgcutout(ra, dec, mesh, line = 0):
    import requests
    import numpy as np
    from PIL import Image
    from PIL import ImageOps
    import io
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry

    retry_strategy = Retry(
        total=20,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    print("\rDownloading image at ra: " + str(ra) + " dec: " + str(dec) + " line: " + str(line), end=' ', flush=True)
    url = r"http://skyserver.sdss.org/dr15/SkyServerWS/ImgCutout/getjpeg?ra=" + str(ra) + r"&dec=" + str(dec) + r"&scale=0.8&height=" + str(mesh) + "&width=" + str(mesh)
    res = http.get(url, timeout=10)
    
    img = Image.open(io.BytesIO(res.content))
    img = ImageOps.grayscale(img)
    img = img.resize((mesh, mesh))
    img = np.array(img)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    #np.savetxt("img\\" + str(line) + ".txt", img)
    return img

def get_SDSS_urls(df, mesh):
    df['urls'] = r"http://skyserver.sdss.org/dr15/SkyServerWS/ImgCutout/getjpeg?ra=" + df['RA'].astype(str) + r"&dec=" + df['DEC'].astype(str) + r"&scale=0.8&height=100&width=100"

def import_simulation_data(sim_dir, mesh, save_dir="..\\data\\fits_training\\", name="fd=0.3-0.7_discs.fits"):
    import pandas as pd
    import glob
    import os
    from joblib import Parallel, delayed
    import numpy as np
    def import_simulation_data_part(path):
        image_batch = pd.read_csv(path, header=None).to_numpy().reshape((-1, mesh, mesh))
        #df = pd.DataFrame(columns=['image'])
        #df['image'] = image_batch.tolist()
        return image_batch
    E = np.concatenate(Parallel(n_jobs=8, prefer="threads")(delayed(import_simulation_data_part)(path) for path in glob.glob(os.path.join(sim_dir, 'no_disc', '**\\', '2dft.dat'), recursive=True)), axis = 0)
    ES = np.concatenate(Parallel(n_jobs=8, prefer="threads")(delayed(import_simulation_data_part)(path) for path in glob.glob(os.path.join(sim_dir, 'disc', '**\\', '2dft.dat'), recursive=True)), axis = 0)
    print(E.shape)
    print(ES.shape)
    print('Finished Concatenating')
    test = np.ones(shape=(E.shape[0]))
    test2 = np.ones(shape=(ES.shape[0]))
    df_E = pd.DataFrame(data=test)
    df_ES = pd.DataFrame(data=test2)
    E = pd.DataFrame(df_E.apply(lambda row: E[row.name], axis=1, raw=False), columns=['image'])
    ES = pd.DataFrame(df_ES.apply(lambda row: ES[row.name], axis=1, raw=False), columns=['image'])

    print('Finished Converting to Dataframe')
    E['class'] = 'E'
    ES['class'] = 'ES'
    df = pd.concat([E, ES], ignore_index=True)
    print(df)
    df['name'] = 'sim'
    del E
    del ES

    #print(df.info())
    #print(df['class'].unique())
    print(type(df))
    print("Dataframe to hdf5")
    galaxy_dataframe_to_hdf5(df, mesh, save_dir, name)


def galaxy_dataframe_to_hdf5(galaxies, mesh, filepath, name):
    import pandas as pd
    import os
    import numpy as np
    import h5py
    from h5py import Group
    hdf5_path = os.path.abspath(os.path.join(filepath, name))
    print(galaxies)
    print(galaxies['image'])
    galaxies.reset_index(drop=True, inplace=True)
    with h5py.File(hdf5_path, "w") as f:

        image_data = np.concatenate(galaxies['image'], axis=0).reshape(-1, mesh, mesh)
        
        print(image_data.shape)
        f.create_dataset("image_data", data=image_data)
    
    galaxies[galaxies.columns.difference(['image'])].to_hdf(hdf5_path, 'metadata', complevel=5, format='table', data_columns=True, mode='a')
