import os
from utils.dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderVal_derain


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)

def get_validation_derain(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_derain(rgb_dir, img_options)
