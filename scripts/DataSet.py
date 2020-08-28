import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import cv2
from tqdm import tqdm
from joblib import Parallel, delayed

upperbody = list(range(0, 6))
lowerbody = [6, 7, 8]
wholebody = list(range(9, 13))
head = [13, 14, 15]
neck = [16]
arms_and_hands = [17, 18]
waist = [19]
legs_and_feet = list(range(20, 24))
others = [24, 25, 26]
garment_parts = list(range(27, 34))
clousures = [34, 35]
decorations = list(range(36, 46))


class DataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        stage,
        csv_info,
        transforms,
        resize_shape=(128, 128),
        train=True,
        cnt=None,
    ):
        self.root = root
        self.stage = stage
        self.transforms = transforms
        self.csv_info = csv_info
        self.resize_shape = resize_shape
        self.imgs = list(sorted(os.listdir(os.path.join(root, stage))))
        if cnt:
            self.imgs = self.imgs[:cnt]

        self.train = train
        if self.train:
            print("Reading imgs in RAM...")
            self.imgs_and_masks = Parallel(n_jobs=6, prefer="threads")(
                delayed(self.get_resized_img_and_mask)(x)
                for x in tqdm(range(len(self.imgs)))
            )
            print("Images read!")

    def get_image(self, idx):
        img_path = os.path.join(self.root, self.stage, self.imgs[idx])
        return Image.open(img_path).convert("RGB")

    def get_resized_img_and_mask(self, idx, return_og_img=False):
        og_img = self.get_image(idx)
        img = og_img.resize(self.resize_shape, Image.NEAREST)

        if not self.train:
            return img, None, og_img
        else:
            np_img = np.asarray(og_img)
            mask = self.get_mask_color_coded(
                self.imgs[idx], np_img.shape[0], np_img.shape[1]
            )
            mask = np.array(mask)
            mask = DataSet.resize_mask(mask, self.resize_shape)
            return img, mask

    @staticmethod
    def resize_mask(mask, shape):
        return cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)

    def __getitem__(self, idx):
        if self.train:
            img, mask = self.imgs_and_masks[idx]
        else:
            img, mask, og_img = self.get_resized_img_and_mask(idx)

        if self.train:
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[1:]
            masks = mask == obj_ids[:, None, None]
            num_objs = len(obj_ids)

            boxes = []

            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1]) + 1
                ymin = np.min(pos[0])
                ymax = np.max(pos[0]) + 1
                if xmin == xmax or ymin == ymax:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])

            # if len(boxes) == 0:
            #     boxes.append([0, 1, 0, 1])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(obj_ids, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            try:
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            except Exception:
                area = torch.Tensor(0)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
        else:
            target = {}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.train:
            return img, target
        else:
            return img, target, og_img

    def __len__(self):
        return len(self.imgs)

    def rleToMask(self, rleString, height, width):
        rows, cols = height, width
        rleNumbers = [int(numstring) for numstring in rleString.split(" ")]
        rlePairs = np.array(rleNumbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)

        for index, length in rlePairs:
            index -= 1
            img[index : index + length] = 255

        img = img.reshape(cols, rows)
        img = img.T

        return img

    def get_mask_color_coded(self, image_name, height, width):
        curr_image = self.csv_info.loc[self.csv_info.ImageId == image_name]
        base_mask = np.zeros(shape=(height, width))

        for mask_idx in range(curr_image.shape[0]):
            mask_row = curr_image.iloc[mask_idx, :]
            class_id = int(mask_row["ClassId"])
            mask_rle_encoding = mask_row["EncodedPixels"]
            mask = self.rleToMask(mask_rle_encoding, height, width)

            if class_id in upperbody:
                base_mask = np.where(mask == 255, 1, base_mask)
            elif class_id in lowerbody:
                base_mask = np.where(mask == 255, 2, base_mask)
            elif class_id in wholebody:
                base_mask = np.where(mask == 255, 3, base_mask)
            elif class_id in head:
                base_mask = np.where(mask == 255, 4, base_mask)
            elif class_id in neck:
                base_mask = np.where(mask == 255, 5, base_mask)
            elif class_id in arms_and_hands:
                base_mask = np.where(mask == 255, 6, base_mask)
            elif class_id in waist:
                base_mask = np.where(mask == 255, 7, base_mask)
            elif class_id in legs_and_feet:
                base_mask = np.where(mask == 255, 8, base_mask)
            elif class_id in others:
                base_mask = np.where(mask == 255, 9, base_mask)
            elif class_id in garment_parts:
                base_mask = np.where(mask == 255, 10, base_mask)
            elif class_id in clousures:
                base_mask = np.where(mask == 255, 11, base_mask)
            elif class_id in decorations:
                base_mask = np.where(mask == 255, 12, base_mask)

        return base_mask.astype(np.uint8)
