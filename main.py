# ------------------------------------------------------------------
# CoMet: Towards Real Unsupervised Anomaly Detection Via Confident Meta-Learning
# ------------------------------------------------------------------

import logging
import os
import sys

import click
import numpy as np
import torch

sys.path.append("src")
import backbones
import common
import metrics
import utils
import comet

LOGGER = logging.getLogger(__name__)


def load_augmented_data(augmented_dir, batch_size, resize=329, imagesize=288):
    """Load pre-generated anomalous images for training."""
    if not augmented_dir or not os.path.exists(augmented_dir):
        return None

    img_dir = os.path.join(augmented_dir, "imgs")
    mask_dir = os.path.join(augmented_dir, "masks")

    if not os.path.exists(img_dir):
        return None

    from torchvision import transforms
    from PIL import Image

    transform_img = transforms.Compose([
        transforms.Resize((int(resize * 2.5 + 0.5), resize)),
        transforms.CenterCrop((int(imagesize * 2.5 + 0.5), imagesize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((int(resize * 2.5 + 0.5), resize)),
        transforms.CenterCrop((int(imagesize * 2.5 + 0.5), imagesize)),
        transforms.ToTensor(),
    ])

    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, img_dir, mask_dir):
            self.img_dir = img_dir
            self.mask_dir = mask_dir
            self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, idx):
            img_name = self.img_files[idx]
            img_path = os.path.join(self.img_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name)

            image = Image.open(img_path).convert("RGB")
            image = transform_img(image)

            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                mask = transform_mask(mask)
            else:
                mask = torch.ones([1, *image.size()[1:]])

            return {
                "image": image,
                "mask": mask,
                "is_anomaly": True,
                "image_path": img_path,
            }

    dataset = AugmentedDataset(img_dir, mask_dir)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )


_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
    "visa": ["datasets.visa", "VisaDataset"],
    "sdd2": ["datasets.sdd2", "SDD2Dataset"],
    "pothole": ["datasets.pothole", "PotholeDataset"]
}


@click.group(chain=True)
@click.option("--results_path", type=str, required=True)
@click.option("--gpu", type=int, default=[0], multiple=True)
@click.option("--seed", type=int, default=0)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", is_flag=True)
@click.option("--save_segmentation_images", is_flag=True, default=False)
@click.option("--augmented_path", type=str, default='')
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test,
    save_segmentation_images,
    augmented_path
):
    methods = dict(methods)
    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    device = utils.set_torch_device(gpu)
    dataloaders_list = methods["get_dataloaders"](seed)

    results = []
    for dataloader_count, dataloaders in enumerate(dataloaders_list):
        dataset_name = dataloaders["training"].name
        LOGGER.info(f"Evaluating dataset [{dataset_name}] ({dataloader_count + 1}/{len(dataloaders_list)})")

        utils.fix_seeds(seed, device)
        imagesize = dataloaders["training"].dataset.imagesize
        comet_list = methods["get_comet"](imagesize, device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)

        for i, comet_model in enumerate(comet_list):
            if comet_model.backbone.seed is not None:
                utils.fix_seeds(comet_model.backbone.seed, device)

            comet_model.set_model_dir(os.path.join(models_dir, str(i)), dataset_name)

            if not test:
                augmented_loader = None
                if augmented_path:
                    augmented_loader = load_augmented_data(
                        augmented_path,
                        dataloaders["training"].batch_size,
                        resize=329,
                        imagesize=288
                    )
                i_auroc, p_auroc, pro = comet_model.train(
                    dataloaders["training"],
                    dataloaders["testing"],
                    augmented_loader
                )
            else:
                i_auroc, p_auroc, pro = comet_model.test(
                    dataloaders["training"],
                    dataloaders["testing"],
                    save_segmentation_images
                )

            result = {
                "dataset_name": dataset_name,
                "instance_auroc": i_auroc,
                "full_pixel_auroc": p_auroc,
                "anomaly_pixel_auroc": pro,
            }
            results.append(result)

            for key, val in result.items():
                if key != "dataset_name":
                    LOGGER.info(f"{key}: {val:.3f}")

        LOGGER.info("-----\n")

    # Save final results
    metric_names = [k for k in results[-1].keys() if k != "dataset_name"]
    dataset_names = [r["dataset_name"] for r in results]
    scores = [list(r.values())[1:] for r in results]

    utils.compute_and_store_final_results(
        run_save_path,
        scores,
        column_names=metric_names,
        row_names=dataset_names,
    )


@main.command("net")
@click.option("--backbone_names", "-b", type=str, multiple=True, default=["wide_resnet50_2"])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=["layer2", "layer3"])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--embedding_size", type=int, default=256)
@click.option("--meta_epochs", type=int, default=40)
@click.option("--aed_meta_epochs", type=int, default=1)
@click.option("--gan_epochs", type=int, default=4)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=None)
@click.option("--noise_std", type=float, default=0.05)
@click.option("--dsc_margin", type=float, default=0.8)
@click.option("--dsc_lr", type=float, default=0.0002)
@click.option("--auto_noise", type=float, default=0)
@click.option("--train_backbone", is_flag=True)
@click.option("--cos_lr", is_flag=True)
@click.option("--pre_proj", type=int, default=1)
@click.option("--proj_layer_type", type=int, default=0)
@click.option("--mix_noise", type=int, default=1)
def net(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    embedding_size,
    meta_epochs,
    aed_meta_epochs,
    gan_epochs,
    noise_std,
    dsc_layers,
    dsc_hidden,
    dsc_margin,
    dsc_lr,
    auto_noise,
    train_backbone,
    cos_lr,
    pre_proj,
    proj_layer_type,
    mix_noise,
):
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_coll = [[] for _ in backbone_names]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer_name = ".".join(layer.split(".")[1:])
            layers_coll[idx].append(layer_name)
    else:
        layers_coll = [layers_to_extract_from]

    def get_comet(input_shape, device):
        models = []
        for name, layers in zip(backbone_names, layers_coll):
            seed = None
            if ".seed-" in name:
                name, seed = name.split(".seed-")[0], int(name.split("-")[-1])
            backbone = backbones.load(name)
            backbone.name, backbone.seed = name, seed

            model = comet.CoMet(device)
            model.load(
                backbone=backbone,
                layers_to_extract_from=layers,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                embedding_size=embedding_size,
                meta_epochs=meta_epochs,
                aed_meta_epochs=aed_meta_epochs,
                gan_epochs=gan_epochs,
                noise_std=noise_std,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
                auto_noise=auto_noise,
                train_backbone=train_backbone,
                cos_lr=cos_lr,
                pre_proj=pre_proj,
                proj_layer_type=proj_layer_type,
                mix_noise=mix_noise,
            )
            models.append(model)
        return models

    return ("get_comet", get_comet)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1.0)
@click.option("--batch_size", default=2, type=int)
@click.option("--num_workers", default=8, type=int)
@click.option("--resize", default=256, type=int)
@click.option("--imagesize", default=224, type=int)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0.0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--augment", is_flag=True)
def dataset(
    name, data_path, subdatasets, train_val_split, batch_size, num_workers,
    resize, imagesize, rotate_degrees, translate, scale, brightness,
    contrast, saturation, gray, hflip, vflip, augment
):
    dataset_info = _DATASETS[name]
    dataset_lib = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for sub in subdatasets:
            train_dataset = dataset_lib.__dict__[dataset_info[1]](
                data_path, classname=sub, resize=resize, train_val_split=train_val_split,
                imagesize=imagesize, split=dataset_lib.DatasetSplit.TRAIN, seed=seed,
                rotate_degrees=rotate_degrees, translate=translate, scale=scale,
                brightness_factor=brightness, contrast_factor=contrast,
                saturation_factor=saturation, gray_p=gray, h_flip_p=hflip,
                v_flip_p=vflip, augment=augment
            )
            test_dataset = dataset_lib.__dict__[dataset_info[1]](
                data_path, classname=sub, resize=resize, imagesize=imagesize,
                split=dataset_lib.DatasetSplit.TEST, seed=seed
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, prefetch_factor=2
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, prefetch_factor=2
            )

            train_loader.name = f"{name}_{sub}" if sub else name
            dataloader_dict = {
                "training": train_loader,
                "testing": test_loader,
                "validation": None
            }
            if train_val_split < 1:
                val_dataset = dataset_lib.__dict__[dataset_info[1]](
                    data_path, classname=sub, resize=resize, train_val_split=train_val_split,
                    imagesize=imagesize, split=dataset_lib.DatasetSplit.VAL, seed=seed
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True
                )
                dataloader_dict["validation"] = val_loader

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()