class_mappings = {
        'E': 0,
        'ES': 1,
        'U': -1
}

def load_fits_data(filepath=".\\califa.fits"):
        from astropy.io import fits
        from astropy.table import Table
        import numpy as np
        with fits.open(filepath) as hdul:
                x = []
                Y = []
                metadata = Table(hdul[1].data).to_pandas()
                x, Y = zip(*[(np.array(hdu.data), class_mappings[hdu.header['CLASS']]) for hdu in hdul if type(hdu) is fits.CompImageHDU or type(hdu) is fits.ImageHDU])
        return x, Y, metadata
