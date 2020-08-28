from DataSet import DataSet
from Model import Model
from transfomations import get_transform
from engine import train_one_epoch, evaluate
import utils
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter
import time


def write_result(og_img, img, prediction, index, epoch, resolution):
    try:
        mask = np.zeros(img.shape[1:])
        # breakpoint()
        try:
            masks = prediction[0]["masks"]
            labels = prediction[0]["labels"].cpu()
        except Exception:
            masks = prediction["masks"]
            labels = prediction["labels"].cpu()
        for x in range(masks.shape[0] - 1, 0, -1):
            if len(masks.shape) == 4:
                tmp_mask = masks[x, 0].mul(255).byte().cpu().numpy()
            else:
                tmp_mask = masks[x].mul(255).byte().cpu().numpy()
            mask = np.where(tmp_mask > 0, labels[x].item(), mask)

        og_mask = DataSet.resize_mask(mask, og_img.size)
    except Exception:
        pass

    mask = Image.fromarray(np.uint8(cm.nipy_spectral(mask / mask.max()) * 255)).convert(
        "RGB"
    )
    og_mask = Image.fromarray(
        np.uint8(cm.nipy_spectral(og_mask / og_mask.max()) * 255)
    ).convert("RGB")
    filepath = Path(f"./results/{epoch}/{resolution}/")
    filepath.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    img.save(filepath / f"img_{index}.png")
    og_img.save(filepath / f"og_img_{index}.png")
    mask.save(filepath / f"mask_{index}.png")
    og_mask.save(filepath / f"og_mask_{index}.png")

    blended = Image.blend(og_img, og_mask, 0.5)
    blended.save(filepath / f"blended_{index}.png")

    og_mask_np = np.asarray(og_mask).astype(np.uint8).transpose(2, 0, 1)
    og_img_np = np.asarray(og_img).astype(np.uint8)
    # breakpoint()
    for x in range(1, og_mask_np.max() + 1):
        submask = (og_mask_np[0] == x).astype(np.uint8)
        if submask.sum() == 0:
            continue

        contours, hierarchy = cv2.findContours(
            submask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        r = x % 3
        g = (x // 3) % 3
        b = (x // 3 // 3) % 3

        for contour in contours:
            cv2.drawContours(og_img_np, contour, -1, (r * 100, g * 100, b * 100), 5)

    cv2.imwrite(str(filepath / f"og_img_{index}_cnts.png"), og_img_np[:, :, ::-1])


def execute(resize_shape, batch_size, num_epochs, write_results=True):
    df = pd.read_csv("./imaterialist-fashion-2019-FGVC6/train.csv")
    dataset = DataSet(
        "./imaterialist-fashion-2019-FGVC6",
        "train",
        df,
        get_transform(train=True),
        resize_shape=resize_shape,
        train=True,
    )
    dataset_test_epoch = DataSet(
        "./imaterialist-fashion-2019-FGVC6",
        "test",
        df,
        get_transform(train=False),
        resize_shape=resize_shape,
        train=False,
        cnt=50,
    )
    dataset_test = DataSet(
        "./imaterialist-fashion-2019-FGVC6",
        "test",
        df,
        get_transform(train=False),
        resize_shape=resize_shape,
        train=False,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=utils.collate_fn,
    )

    data_loader_test_epoch = torch.utils.data.DataLoader(
        dataset_test_epoch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        collate_fn=utils.collate_fn,
    )

    tb_writer = {
        "writer": SummaryWriter(
            log_dir=f"runs/MRCNN-SH:{resize_shape}-BS:{batch_size}-{time.time()}"
        ),
        "step": 0,
    }

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 13
    output_dir = "models"
    utils.mkdir("models")
    model_instance = Model()
    model = model_instance.get_instance_segmentation_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            epoch,
            print_freq=100,
            tb_writer=tb_writer,
        )

        utils.save_on_master(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(output_dir, "model_{}.pth".format(epoch)),
        )

        lr_scheduler.step()
        # put the model in evaluation mode
        model.eval()
        if write_results:
            index = 0
            for _, (imgs, _, og_imgs) in enumerate(tqdm(data_loader_test_epoch)):
                with torch.no_grad():
                    predictions = model([img.to(device) for img in imgs])

                for img, prediction, og_img in zip(imgs, predictions, og_imgs):
                    write_result(og_img, img, prediction, index, epoch, resize_shape)
                    index += 1


def main():
    shapes = [(128, 128)]
    batch_sizes = [8]

    for shape, batch_size in zip(shapes, batch_sizes):
        execute(shape, batch_size, 10, True)


if __name__ == "__main__":
    main()
