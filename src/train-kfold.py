import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from model import WSISurvivalModel
from metrics import calculate_cindex, combined_cox_ranking_loss, negative_partial_log
from custom_dataset import WSISurvivalDataset
from config import IMG_SIZE, BATCH, LR, EPOCHS, PATCH_DIR, NUM_PATCHES, data_comb
from tqdm import tqdm
import timm
from torchinfo import summary
from transformers import AutoImageProcessor, ViTModel
# from huggingface_hub import login
# from conch.open_clip_custom import create_model_from_pretrained

# from resnet import resnet200
# from torchvision.models import resnet200


def train_fold(model, train_loader, val_loader, optimizer, device, num_epochs):
    best_val_cindex = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = negative_partial_log(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            train_cindex = calculate_cindex(model, train_loader, device)
            val_cindex = calculate_cindex(model, val_loader, device)

        train_loss /= len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, '
              f'Train C-index: {train_cindex:.4f}, Val C-index: {val_cindex:.4f}')

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex

    return best_val_cindex

def main():
    # login()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    pretrained_model = timm.create_model(
            # 'hf-hub:1aurent/vit_small_patch16_224.transpath_mocov3',
            "hf-hub:MahmoodLab/uni",
            # 'hf-hub:1aurent/resnet18.tiatoolbox-kather100k',
            # "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
            num_classes=0,
            init_values=1e-5,
            dynamic_img_size=True
            )

    # pretrained_model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="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")

    # pretrained_model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    # transform = AutoImageProcessor.from_pretrained("owkin/phikon")
    pretrained_model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    # transform = create_transform(**resolve_data_config(pretrained_model.pretrained_cfg, model=pretrained_model))
    summary(WSISurvivalModel(pretrained_model), (BATCH, NUM_PATCHES, 3, *IMG_SIZE))

    # dic = {
    #     'sample_input_D': 3,
    #     'sample_input_H': 128,
    #     'sample_input_W': 128,
    #     'num_seg_classes':0
    # }

    # pretrained_model = resnet200(**dic)
    # pretrained_model.load_state_dict(torch.load('/home/ubuntu/Documents/survival analysis/dataset/MedicalNet_pytorch_files2/pretrain/resnet_200.pth')['state_dict'])

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = WSISurvivalDataset(data_comb, transform=transform)

    k_folds = 4
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold+1}/{k_folds}')
        print('--------------------------------')

        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subsampler, batch_size=BATCH, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subsampler, batch_size=BATCH, shuffle=False, num_workers=2)

        model = WSISurvivalModel(pretrained_model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_cindex = train_fold(model, train_loader, val_loader, optimizer, device, EPOCHS)
        fold_results.append(best_val_cindex)

        print(f'Best validation C-index for fold {fold+1}: {best_val_cindex:.4f}')
        print('--------------------------------')

    print('\nK-Fold Cross-Validation Results:')
    print('--------------------------------')
    print(f'Average validation C-index: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f})')
    print(f'Individual fold results: {fold_results}')

if __name__ == '__main__':
    main()
