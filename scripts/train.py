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

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()


def write_result(img, prediction, index):
    try:
        mask = np.zeros(img.shape[1:])
        # breakpoint()
        masks = prediction[0]["masks"]
        labels = prediction[0]["labels"].cpu()
        for x in range(masks.shape[0] - 1, 0, -1):
            tmp_mask = masks[x, 0].mul(255).byte().cpu().numpy()
            mask = np.where(tmp_mask > 0, labels[x].item(), mask)

        # mask = Image.fromarray(
        #     prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy()
        # )
    except Exception:
        breakpoint()

    mask = Image.fromarray(mask)
    filepath = Path("./results/")
    filepath.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    plt.imsave(filepath / f"og_img_{index}.png", img)
    plt.imsave(filepath / f"mask_{index}.png", mask)
    # mask = mask.astype(np.uint8)
    # img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     cv2.drawContours(img, contour, -1, (0, 255, 0), 2)

    # plt.imsave(filepath / "mask_over_orig.png", img)


def main():

    df = pd.read_csv("./imaterialist-fashion-2019-FGVC6/train.csv")
    dataset = DataSet(
        "./imaterialist-fashion-2019-FGVC6", "train", df, get_transform(train=True)
    )
    dataset_test = DataSet(
        "./imaterialist-fashion-2019-FGVC6", "test", df, get_transform(train=False)
    )

    batch_size = 8

    # indices = torch.randperm(len(dataset)).tolist()
    # dataset = torch.utils.data.Subset(dataset, indices[:-50])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=utils.collate_fn,
    # )

    # print("Checking dataset...")
    # for idx, (x, y) in enumerate(tqdm(data_loader)):
    #     pass

    # print("Dataset checked!")
    # breakpoint()

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
    num_epochs = 5

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        utils.save_on_master(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(output_dir, "model_{}.pth".format(epoch)),
        )

        # writer.add_scalar('Loss/train', np.random.random(), n_iter)
        # writer.add_scalar('Loss/test', np.random.random(), n_iter)
        # writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        # writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

        lr_scheduler.step()
        # evaluate after every epoch
        # evaluate(model, data_loader_test, device=device)

        # img = Image.open(
        #     "./imaterialist-fashion-2019-FGVC6/test/0a4aae5ecd970a120bfcc6b377b6e187.jpg"
        # )
        # img = img.resize((600, 600), Image.NEAREST)
        # img = np.array(img) / 255
        # img = img.transpose(2, 0, 1)

        # model.eval()
        # with torch.no_grad():
        #     prediction = model([torch.tensor(img).type(torch.float).to(device)])

        # masks = prediction[0]["masks"].cpu().numpy()[:3, ...]
        # # img[np.stack((masks[0], masks[0], masks[0]), axis=1)[0, ...] > 0.5] = np.array(
        # #     [1, 0, 0]
        # # )
        # mask = prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy()
        # mask = np.where(mask != 0, 255, 0)

        # filepath = Path(f"./results/")
        # filepath.mkdir(parents=True, exist_ok=True)
        # plt.imsave(filepath / "mask.png", mask, cmap="gray")
        # mask = mask.astype(np.uint8)

        # img = img.transpose(1, 2, 0)
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # for contour in contours:
        #     cv2.drawContours(img, contour, -1, (0, 1, 0), 2)

        # plt.imsave(filepath / "mask_over_orig.png", img)
        for index, (img, target) in enumerate(dataset_test):
            breakpoint()
            # img, _ = dataset_test[0]
            # put the model in evaluation mode
            model.eval()
            with torch.no_grad():
                prediction = model([img.to(device)])

            # breakpoint()

            try:
                mask = np.zeros(img.shape[1:])
                # breakpoint()
                masks = prediction[0]["masks"]
                labels = prediction[0]["labels"].cpu()
                for x in range(masks.shape[0] - 1, 0, -1):
                    tmp_mask = masks[x, 0].mul(255).byte().cpu().numpy()
                    mask = np.where(tmp_mask > 0, labels[x].item(), mask)

                # mask = Image.fromarray(
                #     prediction[0]["masks"][0, 0].mul(255).byte().cpu().numpy()
                # )
            except Exception:
                breakpoint()
            mask = Image.fromarray(mask)
            filepath = Path(f"./results/{epoch}/")
            filepath.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            plt.imsave(filepath / f"og_img_{index}.png", img)
            plt.imsave(filepath / f"mask_{index}.png", mask)
    # mask = mask.astype(np.uint8)
    # img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     cv2.drawContours(img, contour, -1, (0, 255, 0), 2)

    # plt.imsave(filepath / "mask_over_orig.png", img)


if __name__ == "__main__":
    main()
