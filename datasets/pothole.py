import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

# All pothole dataset categories
_CLASSNAMES = [
    "cracks-and-potholes-in-road",
    "pothole600",
    "edmcrack600",
    "cnr-road-dataset",
    "gaps384",
    "crack500",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PotholeDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Pothole datasets.
    """

    def __init__(
        self,
        source,
        classname=None,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the pothole data folder.
            classname: [str or None]. Name of pothole dataset class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit.
            train_val_split: [float]. If < 1.0, splits training data into
                             training and validation.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # Define image transformations
        self.transform_img = [
            transforms.Resize(resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees, 
                                    translate=(translate, translate),
                                    scale=(1.0-scale, 1.0+scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        # Define mask transformations
        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        
        # Handle different image formats (jpg, png)
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # Load mask if in test mode and an anomaly sample
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            
            # Skip if class doesn't exist
            if not os.path.isdir(classpath):
                continue
                
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                
                # Skip if not a directory
                if not os.path.isdir(anomaly_path):
                    continue
                    
                anomaly_files = sorted(os.listdir(anomaly_path))
                
                # Filter only image files (jpg, png)
                anomaly_files = [f for f in anomaly_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                # Handle train/val split if needed
                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                # Handle mask paths for test anomalies
                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    
                    # Skip if mask directory doesn't exist
                    if not os.path.isdir(anomaly_mask_path):
                        maskpaths_per_class[classname][anomaly] = [None] * len(imgpaths_per_class[classname][anomaly])
                        continue
                        
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    # Filter only image files
                    anomaly_mask_files = [f for f in anomaly_mask_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    
                    # Handle cases where filenames might differ between test and ground truth
                    if len(anomaly_mask_files) != len(imgpaths_per_class[classname][anomaly]):
                        # Try to match filenames without extensions
                        test_basenames = [os.path.splitext(os.path.basename(p))[0] for p in imgpaths_per_class[classname][anomaly]]
                        mask_basenames = [os.path.splitext(f)[0] for f in anomaly_mask_files]
                        
                        maskpaths = []
                        for test_file in test_basenames:
                            found = False
                            for i, mask_file in enumerate(mask_basenames):
                                if test_file == mask_file:
                                    maskpaths.append(os.path.join(anomaly_mask_path, anomaly_mask_files[i]))
                                    found = True
                                    break
                            if not found:
                                maskpaths.append(None)  # No matching mask found
                        
                        maskpaths_per_class[classname][anomaly] = maskpaths
                    else:
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                else:
                    if anomaly == "good":
                        maskpaths_per_class[classname]["good"] = None

        # Unroll the data dictionary to an easy-to-iterate list
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        # Handle case where maskpaths_per_class[classname][anomaly] might be None
                        if anomaly in maskpaths_per_class.get(classname, {}) and maskpaths_per_class[classname][anomaly] is not None:
                            if i < len(maskpaths_per_class[classname][anomaly]):
                                data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                            else:
                                data_tuple.append(None)
                        else:
                            data_tuple.append(None)
                    else:
                        data_tuple.append(None)
                    
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate