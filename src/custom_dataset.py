import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from PIL import Image
from config import *


class WSISurvivalDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index()
        self.patch_size = IMG_SIZE[0]
        self.transform = transform
        self.selected_files = self.get_patch()

    def get_patch(self):
        select_dict = dict()

        for _, row in self.dataframe.iterrows():
            if np.uint8(row['Overall Survival Status'][0]) == 1:
                wsi_path = f'{PATCH_DIR}/dead/{row["Patient Identifier"]}'
            else:
                wsi_path = f'{PATCH_DIR}/living/{row["Patient Identifier"]}'

            patch_files = os.listdir(wsi_path)

            if len(patch_files) > NUM_PATCHES:
                patch_files = np.random.choice(
                    patch_files, NUM_PATCHES, replace=False)
            else:
                patch_files = np.random.choice(
                    patch_files, NUM_PATCHES, replace=True)

            select_dict[row['Patient Identifier']] = patch_files

        return select_dict

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_data = self.dataframe.iloc[idx]
        time = patient_data['Overall Survival (Months)']
        event = np.uint8(patient_data['Overall Survival Status'][0])

        if event == 1:
            wsi_path = f'{PATCH_DIR}/dead/{patient_data["Patient Identifier"]}'
        else:
            wsi_path = f'{PATCH_DIR}/living/{patient_data["Patient Identifier"]}'

        patches = self._load_wsi_patches(
            self.selected_files[patient_data['Patient Identifier']], wsi_path)
        patches_tensor = torch.stack(
            [self._process_patch(patch) for patch in patches])
        
        label = torch.tensor([time, event], dtype=torch.float)

        return patches_tensor, label

    def _load_wsi_patches(self, patch_files, wsi_path):
        patches = []
        for patch_file in patch_files:
            patch_path = os.path.join(wsi_path, patch_file)
            patch = Image.open(patch_path).convert('RGB')
            patches.append(patch.resize(IMG_SIZE, Image.LANCZOS))

        return patches

    def _process_patch(self, patch):
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = np.array(patch)
            patch = patch.transpose((2, 0, 1))
            patch = torch.from_numpy(patch).float() / 255.0

        return patch
