import glob
import random
from skimage import io, transform
from torch.utils import data
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize
import warnings

class PolyUDataset(data.Dataset):
    DATASET_BASE_DIR = '../../data/PolyU-Real-World-Noisy-Images-Dataset/'
    DATASET_SEED = '84239482109'
    DATA_SPLIT_PERCENTS = {'train' : (0.0, 0.66),
                           'val' : (0.66, 0.83),
                           'test' : (0.83, 1.0),
                           'all' : (0.0, 1.0)}
                           
    def __init__(self, split_type='train', use_cropped_images=True, image_type='real',
                 in_memory=True, dataset_dir=DATASET_BASE_DIR, downsample_shape=None,
                 as_grayscale=False):
        """
            Args:
                split_type: Whether to retrieve training, validation, or testing data.
                             Possible values: ['train', 'test', 'val', 'all']
                use_cropped_images: If True, retrieves the images cropped to 512 x 512.
                                     If False, all images will have their original size.
                image_type: Whether to retrieve only the "mean", "real", or all images
                             "real" images contain a good amount of noise
                             "mean" images are "real" images except smoothed, probably
                              with an average filter
                             Possible values: ['mean', 'real', 'all']
                in_memory: Whether or not to cache all images in memory 
                dataset_dir: The file path to the directory with the CroppedImages/ and
                              OriginalImages/ subfolders
                downsample_shape: The size tuple to resize the loaded images to. If None
                                   no reshaping will be done on the loaded dataset
                as_grayscale: If True, images will be converted to grayscale
        """
        assert(split_type in ['train', 'test', 'val', 'all'])
        assert(image_type in ['mean', 'real', 'all'])
        if use_cropped_images:
            dataset_dir += 'CroppedImages/'
        else:
            dataset_dir += 'OriginalImages/'
        self.in_memory = in_memory
        self.downsample_shape = downsample_shape
        self.as_grayscale = as_grayscale
        
        # Find only the images labelled with the suffix specified by image_type
        image_file_path_glob = dataset_dir + '*.[jJ][pP][gG]'
        if image_type != 'all':
            image_file_path_glob = dataset_dir + '*_' + image_type + '.[jJ][pP][gG]'        
        
        # Deterministically shuffle the data filepaths to randomly assign images to train, test, or val
        all_image_file_paths = glob.glob(image_file_path_glob)
        random.seed(PolyUDataset.DATASET_SEED)
        random.shuffle(all_image_file_paths)
        
        if len(all_image_file_paths) <= 0:
            warnings.warn(f'No image files found in dataset_dir')
                
        # Partition the data according the the split percents defined at the top of this class
        (data_start_percent, data_end_percent) = PolyUDataset.DATA_SPLIT_PERCENTS[split_type]
        split_start_index = int(data_start_percent * len(all_image_file_paths))
        split_end_index = int(data_end_percent * len(all_image_file_paths))
        self.split_data_file_paths = all_image_file_paths[split_start_index:split_end_index]
        
        # If in_memory is specified, cache all images for the split in memory
        if self.in_memory:
            self.image_data = [io.imread(image_file_path, as_gray=as_grayscale)
                               for image_file_path
                               in self.split_data_file_paths]
            if self.downsample_shape is not None:
                self.image_data = [transform.resize(img, self.downsample_shape) for img in self.image_data]
                    
        
    def __len__(self):
        return len(self.split_data_file_paths)
        
    def __getitem__(self, index):
        if self.in_memory:
            index_image = self.image_data[index]
        else:
            image_file_path = self.split_data_file_paths[index]
            index_image = io.imread(image_file_path, as_gray=as_grayscale)
            
            if self.downsample_shape is not None:
                index_image = transform.resize(index_image, self.downsample_shape)
            
        return ToTensor()(index_image)


if __name__ == "__main__":
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    TEST_BATCH_SIZE = 16
    IMAGE_TYPE = 'mean' # can also be 'real' or 'all'

    # Example dataloaders for training, validation, and testing
    training_dataset = PolyUDataset(split_type='train', image_type=IMAGE_TYPE)
    training_dataloader = data.DataLoader(training_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    print(f'Training: {len(training_dataset)} images')
    for (i, current_image_batch) in enumerate(training_dataloader):
        print(f'Batch {i}: {current_image_batch.shape}')
    print()

    validation_dataset = PolyUDataset(split_type='val', image_type=IMAGE_TYPE, in_memory=False)
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    print(f'Validation: {len(validation_dataset)} images')
    for (i, current_image_batch) in enumerate(validation_dataloader):
        print(f'Batch {i}: {current_image_batch.shape}')
    print()

    test_dataset = PolyUDataset(split_type='test', image_type=IMAGE_TYPE)
    test_dataloader = data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    print(f'Testing: {len(test_dataset)} images')
    for (i, current_image_batch) in enumerate(test_dataloader):
        print(f'Batch {i}: {current_image_batch.shape}')
    print()