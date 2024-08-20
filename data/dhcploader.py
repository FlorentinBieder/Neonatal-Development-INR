import scipy.ndimage
import torch
import torch.nn
import os
import os.path
import nibabel
import pandas as pd

class DHCPDataset(torch.utils.data.Dataset):
    def __init__(self, directory, database, normalize=None,
                 train_ssl=False, cache_images=False):
        '''
        directory is expected to contain the data
        database should be the path to a csv file with columns: index, subject_id, session_id, age, image_path, mask_path
        '''
        super().__init__()
        self.cache_images = cache_images
        self.image_cache = dict()
        self.coord_cache = None   # field for caching the coordinates that we append in every step
        self.directory = os.path.expanduser(directory)
        self.database = pd.read_csv(os.path.expanduser(database), skipinitialspace=True, index_col=0)
        self.normalize = normalize or (lambda x: x)
        dtypes = dict(subject_id=str, session_id=str, age=float, image_path=str, mask_path=str)
        self.database = self.database.astype(dtypes)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, x):
        row = self.database.iloc[x]
        if self.cache_images and x in self.image_cache:
            return self.image_cache[x]

        fp_img = os.path.join(self.directory, row['image_path'])
        fp_mask = os.path.join(self.directory, row['mask_path'])
        try:
            image = nibabel.load(fp_img).get_fdata()
            mask = nibabel.load(fp_mask).get_fdata()
            mask = mask != 0
        except FileNotFoundError:
            print('file not found')
            pass
        # make fit into 192 x 256 x 256 volume, could be anything
        if image.shape != (192, 256, 256):
            zoom = 256/min(image.shape[-1], image.shape[-2])
            image = scipy.ndimage.zoom(input=image, zoom=zoom, order=3)
            mask = scipy.ndimage.zoom(input=mask, zoom=zoom, order=0)
            assert image.shape == (192, 256, 256), 'image shape mismatch'
            assert mask.shape == (192, 256, 256), 'mask shape mismatch'

        image = torch.from_numpy(image[None, ...]) # add artificial channel
        age = torch.tensor(row['age']).float() / 100 # normalize age

        # normalization
        image = self.normalize(image)
        return_object = dict(image=image, mask=mask, age=age)
        return_object.update({k:row[k] for k in row.keys() if k not in return_object})

        if self.cache_images and x not in self.image_cache:
            self.image_cache[x] = return_object
        return return_object