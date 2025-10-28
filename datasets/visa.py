# import os
# from enum import Enum
# import PIL
# import torch
# from torchvision import transforms
# import logging

# LOGGER = logging.getLogger(__name__)

# _CLASSNAMES = [
#     "candle",
#     "capsules",
#     "cashew",
#     "chewinggum",
#     "fryum",
#     "macaroni1",
#     "macaroni2",
#     "pcb1",
#     "pcb2",
#     "pcb3",
#     "pcb4",
#     "pipe_fryum",
# ]

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

# class DatasetSplit(Enum):
#     TRAIN = "train"
#     VAL = "val"
#     TEST = "test"

# class VisaDataset(torch.utils.data.Dataset):
#     """
#     PyTorch Dataset for VisA, matching MVTec interface.
#     """

#     def __init__(
#         self,
#         source,
#         classname,
#         resize=256,
#         imagesize=224,
#         split=DatasetSplit.TRAIN,
#         train_val_split=1.0,
#         rotate_degrees=0,
#         translate=0,
#         brightness_factor=0,
#         contrast_factor=0,
#         saturation_factor=0,
#         gray_p=0,
#         h_flip_p=0,
#         v_flip_p=0,
#         scale=0,
#         **kwargs,
#     ):
#         super().__init__()
#         self.source = source
#         self.split = split
#         self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
#         self.train_val_split = train_val_split
#         self.transform_std = IMAGENET_STD
#         self.transform_mean = IMAGENET_MEAN
        
#         # Get image data
#         self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

#         # Image transforms exactly matching MVTec
#         self.transform_img = [
#             transforms.Resize(resize),
#             transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
#             transforms.RandomHorizontalFlip(h_flip_p),
#             transforms.RandomVerticalFlip(v_flip_p),
#             transforms.RandomGrayscale(gray_p),
#             transforms.RandomAffine(
#                 rotate_degrees, 
#                 translate=(translate, translate),
#                 scale=(1.0-scale, 1.0+scale),
#                 interpolation=transforms.InterpolationMode.BILINEAR
#             ),
#             transforms.CenterCrop(imagesize),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#         ]
#         self.transform_img = transforms.Compose(self.transform_img)

#         # Mask transforms
#         self.transform_mask = [
#             transforms.Resize(resize),
#             transforms.CenterCrop(imagesize),
#             transforms.ToTensor(),
#         ]
#         self.transform_mask = transforms.Compose(self.transform_mask)

#         self.imagesize = (3, imagesize, imagesize)

#     def __getitem__(self, idx):
#         classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
#         image = PIL.Image.open(image_path).convert("RGB")
#         image = self.transform_img(image)

#         if self.split == DatasetSplit.TEST and mask_path is not None:
#             mask = PIL.Image.open(mask_path)
#             mask = self.transform_mask(mask)
#         else:
#             mask = torch.zeros([1, *image.size()[1:]])

#         # anomaly is already either "good" or "anomaly" from get_image_data
#         return {
#             "image": image,
#             "mask": mask,
#             "classname": classname,
#             "anomaly": anomaly,
#             "is_anomaly": int(anomaly == "anomaly"),  # 1 for anomaly, 0 for good
#             "image_name": "/".join(image_path.split("/")[-4:]),
#             "image_path": image_path,
#         }

#     def __len__(self):
#         return len(self.data_to_iterate)

#     def get_image_data(self):
#         imgpaths_per_class = {}
#         maskpaths_per_class = {}

#         for classname in self.classnames_to_use:
#             LOGGER.info(f"Processing class: {classname}")
            
#             class_root = os.path.join(self.source, classname, "Data")
#             images_path = os.path.join(class_root, "Images")
#             masks_path = os.path.join(class_root, "Masks")
            
#             imgpaths_per_class[classname] = {}
#             maskpaths_per_class[classname] = {}

#             # Process normal images
#             normal_path = os.path.join(images_path, "Normal")
#             normal_files = []
            
#             # Handle both upper and lowercase extensions
#             for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
#                 normal_files.extend(sorted([f for f in os.listdir(normal_path) if f.endswith(ext)]))
            
#             LOGGER.info(f"Found {len(normal_files)} normal files")

#             if self.split == DatasetSplit.TEST:
#                 # For test set, include both normal and anomaly
#                 split_idx = int(len(normal_files) * 0.8)
#                 test_normal_files = normal_files[split_idx:]
                
#                 imgpaths_per_class[classname]["good"] = [
#                     os.path.join(normal_path, x) for x in test_normal_files
#                 ]
#                 LOGGER.info(f"Test set normal (good) samples: {len(imgpaths_per_class[classname]['good'])}")
                
#                 # Process anomaly images for test set
#                 anomaly_path = os.path.join(images_path, "Anomaly")
#                 anomaly_mask_path = os.path.join(masks_path, "Anomaly")
                
#                 anomaly_files = []
#                 for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
#                     anomaly_files.extend(sorted([f for f in os.listdir(anomaly_path) if f.endswith(ext)]))
                
#                 LOGGER.info(f"Found {len(anomaly_files)} anomaly files")
                
#                 imgpaths_per_class[classname]["anomaly"] = []
#                 maskpaths_per_class[classname]["anomaly"] = []
                
#                 for img_file in anomaly_files:
#                     img_path = os.path.join(anomaly_path, img_file)
#                     # Get base name without extension and add .png for mask
#                     base_name = os.path.splitext(img_file)[0]
#                     mask_path = os.path.join(anomaly_mask_path, f"{base_name}.png")
                    
#                     if os.path.exists(mask_path):
#                         imgpaths_per_class[classname]["anomaly"].append(img_path)
#                         maskpaths_per_class[classname]["anomaly"].append(mask_path)
#                     else:
#                         LOGGER.warning(f"No mask found for {img_file}")
                
#                 LOGGER.info(f"Test set anomaly samples: {len(imgpaths_per_class[classname]['anomaly'])}")
                
#             else:
#                 # For training, use only normal images
#                 split_idx = int(len(normal_files) * 0.8)
#                 train_normal_files = normal_files[:split_idx]
                
#                 if self.train_val_split < 1.0 and self.split != DatasetSplit.TEST:
#                     val_split_idx = int(len(train_normal_files) * self.train_val_split)
#                     if self.split == DatasetSplit.TRAIN:
#                         train_normal_files = train_normal_files[:val_split_idx]
#                     elif self.split == DatasetSplit.VAL:
#                         train_normal_files = train_normal_files[val_split_idx:]
                
#                 imgpaths_per_class[classname]["good"] = [
#                     os.path.join(normal_path, x) for x in train_normal_files
#                 ]
#                 LOGGER.info(f"Train set normal samples: {len(imgpaths_per_class[classname]['good'])}")
            
#             maskpaths_per_class[classname]["good"] = None

#         # Create final list to iterate over
#         data_to_iterate = []
#         for classname in sorted(imgpaths_per_class.keys()):
#             for anomaly in sorted(imgpaths_per_class[classname].keys()):
#                 for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
#                     data_tuple = [classname, anomaly, image_path]
#                     if self.split == DatasetSplit.TEST and anomaly != "good":
#                         data_tuple.append(maskpaths_per_class[classname][anomaly][i])
#                     else:
#                         data_tuple.append(None)
#                     data_to_iterate.append(data_tuple)

#         # Final debug info
#         LOGGER.info(f"Total samples for {self.split}: {len(data_to_iterate)}")
#         if self.split == DatasetSplit.TEST:
#             n_good = sum(1 for x in data_to_iterate if x[1] == "good")
#             n_anomaly = sum(1 for x in data_to_iterate if x[1] == "anomaly")
#             LOGGER.info(f"Test set composition - Good: {n_good}, Anomaly: {n_anomaly}")

#         return imgpaths_per_class, data_to_iterate

import os
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import PIL
import torch
from torchvision import transforms
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)

# Constants
VISA_CLASSNAMES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
TRAIN_TEST_SPLIT = 0.8  # 80% for training

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class VisaDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for VisA anomaly detection."""

    def __init__(
        self,
        source: str,
        classname: str,
        resize: int = 256,
        imagesize: int = 224,
        split: DatasetSplit = DatasetSplit.TRAIN,
        train_val_split: float = 1.0,
        rotate_degrees: float = 0,
        translate: float = 0,
        brightness_factor: float = 0,
        contrast_factor: float = 0,
        saturation_factor: float = 0,
        gray_p: float = 0,
        h_flip_p: float = 0,
        v_flip_p: float = 0,
        scale: float = 0,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split if isinstance(split, DatasetSplit) else DatasetSplit(split)
        self.classnames_to_use = [classname] if classname is not None else VISA_CLASSNAMES
        self.train_val_split = train_val_split
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.imagesize = (3, imagesize, imagesize)

        # Set up transforms
        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(
                rotate_degrees, 
                translate=(translate, translate),
                scale=(1.0-scale, 1.0+scale),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        # Load dataset
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        LOGGER.info(f"Loaded {len(self.data_to_iterate)} samples for {classname} ({split.value})")

    def create_mask_from_coordinates(self, img_shape: Tuple[int, int, int], coordinates: str) -> np.ndarray:
        """Create binary mask from annotation coordinates."""
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        if not coordinates or coordinates == '[]':
            return mask

        try:
            coords_list = eval(coordinates)
            if isinstance(coords_list, list):
                for polygon in coords_list:
                    if len(polygon) > 2:
                        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(mask, [pts], 255)
        except Exception as e:
            LOGGER.error(f"Error parsing coordinates: {e}")
        return mask

    def get_image_data(self) -> Tuple[Dict, List]:
        """Load and organize dataset images and masks."""
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            # Setup paths
            data_path = os.path.join(self.source, classname, "Data")
            images_path = os.path.join(data_path, "Images")
            masks_path = os.path.join(data_path, "Masks")
            anno_path = os.path.join(self.source, classname, "image_anno.csv")

            # Initialize class dictionaries
            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            # Load annotations if available
            if os.path.exists(anno_path):
                anno_df = pd.read_csv(anno_path, header=None, 
                                    names=['image_path', 'defect_type', 'mask_path'])
            else:
                LOGGER.warning(f"Annotation file not found: {anno_path}")
                continue

            # Process normal images
            normal_path = os.path.join(images_path, "Normal")
            if not os.path.exists(normal_path):
                LOGGER.warning(f"Normal path not found: {normal_path}")
                continue

            # Get normal files and apply splits
            normal_files = sorted([
                f for f in os.listdir(normal_path)
                if f.lower().endswith(VALID_IMG_EXTENSIONS)
            ])

            if self.split == DatasetSplit.TEST:
                # For test set, include all anomaly images and a portion of normal images
                anomaly_path = os.path.join(images_path, "Anomaly")
                anomaly_files = []
                if os.path.exists(anomaly_path):
                    anomaly_files = sorted([
                        f for f in os.listdir(anomaly_path)
                        if f.lower().endswith(VALID_IMG_EXTENSIONS)
                    ])

                # Calculate how many normal samples to include in test set
                # Use same number as anomaly samples to maintain balance
                n_anomaly = len(anomaly_files)
                n_test_normal = max(n_anomaly, len(normal_files) // 5)  # At least 20% of normal
                test_normal_files = normal_files[-n_test_normal:]

                # Add normal samples
                imgpaths_per_class[classname]["good"] = [
                    os.path.join(normal_path, x) for x in test_normal_files
                ]
                maskpaths_per_class[classname]["good"] = [None] * len(test_normal_files)

                # Add anomaly samples
                imgpaths_per_class[classname]["anomaly"] = []
                maskpaths_per_class[classname]["anomaly"] = []

                for img_file in anomaly_files:
                    img_path = os.path.join(anomaly_path, img_file)
                    base_name = os.path.splitext(img_file)[0]

                    # Check if the mask exists with .JPG extension
                    mask_path_jpg = os.path.join(masks_path, "Anomaly", f"{base_name}.JPG")
                    mask_path_png = os.path.join(masks_path, "Anomaly", f"{base_name}.png")

                    if os.path.exists(mask_path_jpg):
                        mask_path = mask_path_jpg  # Use .JPG mask
                    elif os.path.exists(mask_path_png):
                        mask_path = mask_path_png  # Fallback to .png if necessary
                    else:
                        mask_path = None  # No mask found

                    if os.path.exists(img_path) and mask_path is not None:
                        imgpaths_per_class[classname]["anomaly"].append(img_path)
                        maskpaths_per_class[classname]["anomaly"].append(mask_path)


            else:
                # For training, use remaining normal images
                if self.split == DatasetSplit.TRAIN:
                    # Use first 80% of normal files for training
                    n_train = int(len(normal_files) * 0.8)
                    train_files = normal_files[:n_train]
                    if self.train_val_split < 1.0:
                        split_idx = int(len(train_files) * self.train_val_split)
                        train_files = train_files[:split_idx]
                else:  # VAL
                    n_train = int(len(normal_files) * 0.8)
                    train_files = normal_files[:n_train]
                    split_idx = int(len(train_files) * self.train_val_split)
                    train_files = train_files[split_idx:]

                imgpaths_per_class[classname]["good"] = [
                    os.path.join(normal_path, x) for x in train_files
                ]
                maskpaths_per_class[classname]["good"] = [None] * len(train_files)

            LOGGER.info(f"{classname} {self.split.value} set: "
                       f"normal={len(imgpaths_per_class[classname].get('good', []))} "
                       f"anomaly={len(imgpaths_per_class[classname].get('anomaly', []))}")

        # Create final data list
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    mask_path = (maskpaths_per_class[classname][anomaly][i] 
                               if self.split == DatasetSplit.TEST and anomaly != "good"
                               else None)
                    data_tuple = [classname, anomaly, image_path, mask_path]
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample from the dataset."""
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        
        # Load and transform image
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # Load and transform mask if available
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
            # Cleanup temporary mask
            if '_mask.png' in mask_path:
                os.remove(mask_path)
        else:
            mask = torch.zeros([1, *image.shape[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.basename(image_path),
            "image_path": image_path,
        }

    def __len__(self) -> int:
        """Get the total number of samples."""
        return len(self.data_to_iterate)