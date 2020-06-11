import os, sys
import pickle
import numpy as np

def write_to_file(fname, param):
    '''param: (azimuth_deg, elevation_deg, theta_deg, rho)
    '''
    with open(fname, 'wb') as f:
        pickle.dump(param, f)

def read_from_stat_file(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    return lines

def read_exr(filename):
    """Read RGB + Depth data from EXR image file.
    Parameters
    ----------
    filename : str
        File path.
    Returns
    -------
    channels: info in float32 format.
    keys: corresponding keys.
    """
    import OpenEXR as exr
    import Imath
    if filename[-4:] == '.pkl':
        return read_exr_from_pkl(filename)
    exrfile = exr.InputFile(filename)
    keys = exrfile.header()['channels'].keys()
    dw = exrfile.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    channels = []
    for c in list(keys):
        info = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        info = np.fromstring(info, dtype=np.float32)
        info = np.reshape(info, isize)
        channels.append(info)
    return channels, list(keys)

def read_exr_from_pkl(filename):
    with open(filename, 'rb') as f:
        pixels = pickle.load(f)
    keys = pixels.keys()
    channels = []
    for c in keys:
        channels.append(pixels[c])
    return channels, list(keys)

def write_exr(filename, channels, keys):
    write_exr_to_pkl(filename, channels, keys)

def write_exr_to_pkl(filename, channels, keys):
    pixels = {}
    for channel, key in zip(channels, keys):
        pixels[key] = channel
    with open(filename, 'wb') as f:
        pickle.dump(pixels, f)
 
def get_image_file_list(src_folder, dst_folder):
    '''
    get list of image files and makedirs for each fname in the dst folder.
    '''
    image_list = []
    unused_image_list = []
    flist = os.listdir(src_folder)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    # from tqdm import tqdm
    # for fname in tqdm(flist):
    for fname in flist:
        nowpath = os.path.join(src_folder, fname)
        if os.path.isfile(nowpath):
            if fname[-4:] == '.png':
                image_list.append(fname)
            continue
        sub_flist = os.listdir(nowpath)

        output_path = os.path.join(dst_folder, fname)
        if (not os.path.exists(output_path)) and len(sub_flist) != 0:
            os.makedirs(output_path)

        for sub_fname in sub_flist:
            if sub_fname[-4:] != '.png':
                continue
            image_list.append(os.path.join(fname, sub_fname))

    # print('{0} training images in total.'.format(len(image_list)))
    return image_list


