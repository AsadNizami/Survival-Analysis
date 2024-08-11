import torch
import pandas as pd
from torchvision import transforms
from model import WSISurvivalModel
from metrics import cox_loss, calculate_cindex
from custom_dataset import WSISurvivalDataset
from torch.utils.data import DataLoader
from config import IMG_SIZE, BATCH, train_set, val_set, LR, EPOCHS, track, SAVE_PATH, track_dict, TRACKER_CSV_NAME


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = WSISurvivalModel().to(device)
# summary(model, (num_patches, *IMG_SIZE, 3))

train_loader = DataLoader(
    WSISurvivalDataset(train_set, transform=transform),
    batch_size=BATCH, shuffle=True, num_workers=4
)

val_loader = DataLoader(
    WSISurvivalDataset(val_set, transform=transform),
    batch_size=BATCH, shuffle=True, num_workers=4
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_cindex = 0

for epoch in range(EPOCHS):
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs, attention_weights = model(batch_x)
        loss = cox_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_cindex = calculate_cindex(model, train_loader, device)
    val_cindex = calculate_cindex(model, val_loader, device)

    train_loss = train_loss / len(train_loader)

    if val_cindex > best_cindex:
        best_cindex = val_cindex
        torch.save(model.state_dict(), SAVE_PATH)

    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, '
          f'Train C-index: {train_cindex:.4f}, Val C-index: {val_cindex:.4f}')

    track_dict[f'train_C-ind ({epoch+1}/{EPOCHS}))'] = train_cindex
    track_dict[f'val_C-ind ({epoch+1}/{EPOCHS}))'] = val_cindex

    track_dict['Best Train C-ind'] = max(
        track_dict['Best Train C-ind'], round(train_cindex, 3))
        
    track_dict['Best Val C-ind'] = max(
        track_dict['Best Val C-ind'], val_cindex)
    
    track_dict['Least loss'] = min(track_dict['Least loss'], train_loss)

track = pd.concat([track, pd.DataFrame([track_dict])], ignore_index=True)
track.to_csv(TRACKER_CSV_NAME)
