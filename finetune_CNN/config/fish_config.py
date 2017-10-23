# define the paths to the images directory
IMAGES_PATH = "/mnt/dissertation_work/resizedfish/resized224x224AA"

# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 10
NUM_VAL_IMAGES = 200 * NUM_CLASSES
NUM_TEST_IMAGES = 200 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "hdf5/train.hdf5"
VAL_HDF5 = "hdf5/val.hdf5"
TEST_HDF5 = "hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = "output/fish10.model"

# define the path to the dataset mean
DATASET_MEAN = "output/fish_mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"
