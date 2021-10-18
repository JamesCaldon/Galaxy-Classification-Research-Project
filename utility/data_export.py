def fits_to_text(filepath, name, save_dir):
    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    fits_path = os.path.abspath(os.path.join(filepath, name))

    with fits.open(fits_path) as hdul:
        print("Primary Header \n", repr(hdul[0].header))
        print("First BinTable Data as a pandas dataframe \n", repr(Table(hdul[1].data).to_pandas()))
        print("First Comp Image Hdu Header \n", repr(hdul[2].header))
        unique_classes = Table(hdul[1].data).to_pandas()['class'].unique()
        fhandles = {}
        for cl in unique_classes:
            fhandles[str(cl).strip()] = open(os.path.join(save_dir, cl + ".dat"), 'a')
        print(len(hdul))
        for hdu in hdul:
            if type(hdu) is fits.CompImageHDU or type(hdu) is fits.ImageHDU:
                np.savetxt(fhandles[str(hdu.header['CLASS']).strip()], np.array(hdu.data).flatten())

        # Cleanup
        for fh in fhandles:
            fhandles[fh].flush()
            fhandles[fh].close()


def hdf5_to_img(hdf5_name, save_fp):
    import data_loading
    from PIL import Image
    import numpy as np
    import pandas as pd
    x, y, metadata = data_loading.load_hdf5_data(hdf5_name)
    metadata.sa
    classes, class_indices = np.unique(y, return_index=True, axis=0)
    for i in range(len(classes)):
        cls = classes[i]
        cls_index = class_indices[i]
        [Image.fromarray(img).save(save_fp + "/" + cls + "/" + str(i) + ".png") for img in x[cls_index]]

    