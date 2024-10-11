import numpy as np  # type: ignore
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

# Check for GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print({device})
# Load the JSON data
with open("/home/guests2/ghc/New/Tasks/31-LIST-ARC/22eb0ac0_31_full_hor_Massive_list.json", "r") as file:
    data = json.load(file)

class CustomDataset(Dataset):
    def __init__(self, data, max_size):
        self.data = data
        self.max_size = max_size
        self.inputs = []
        self.targets = []
        for item in data:
            input_image = np.array(item['input'])
            target_image = np.array(item['output'], dtype=np.int64)
            height, width = input_image.shape
            
            # Pad the input and target images
            padded_input = np.full((max_size, max_size), 10, dtype=np.float32)  # Use 10 as the padding class
            padded_target = np.full((max_size, max_size), 10, dtype=np.int64)  # Use 10 as the padding class
            padded_input[:height, :width] = input_image
            padded_target[:height, :width] = target_image
            
            self.inputs.append(padded_input)
            self.targets.append(padded_target)
        
        self.inputs = np.array(self.inputs)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32).unsqueeze(0), \
               torch.tensor(self.targets[idx], dtype=torch.int64)

# Parameters
max_size = 30  # Maximum dimension for height or width

# Create dataset and dataloader
train_data = data['train']
test_data = data['test']

train_dataset = CustomDataset(train_data, max_size)
train_dataloader = DataLoader(train_dataset, batch_size=36, shuffle=True)

test_dataset = CustomDataset(test_data, max_size)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = CBR(in_channels, 64)
        self.encoder2 = CBR(64, 128)
        self.encoder3 = CBR(128, 256)
        self.encoder4 = CBR(256, 512)
        self.encoder5 = CBR(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1)
        self.decoder4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1)
        self.decoder3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1)
        self.decoder2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1)
        self.decoder1 = CBR(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)

# Initialize the model
model = UNet(in_channels=1, out_channels=11).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=10)  # Ignore padding class during loss computation
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_dataloader)}')

print('Finished Training')


# Testing loop with visualization
model.eval()
test_loss = 0.0
with torch.no_grad():
    counter = 0
    for inputs, targets in test_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        # Visualize the first batch
        for i in range(inputs.size(0)):
            input_arr = inputs[i].cpu().numpy().squeeze()
            output_arr = torch.argmax(outputs[i], dim=0).cpu().numpy()
            target_arr = targets[i].cpu().numpy()

            # Get the valid region (non-padding) from the input array
            valid_mask = input_arr != 10
            valid_indices = np.where(valid_mask)
            min_row, max_row = valid_indices[0].min(), valid_indices[0].max() + 1
            min_col, max_col = valid_indices[1].min(), valid_indices[1].max() + 1

            # Slice the arrays to remove the padding
            input_valid = input_arr[min_row:max_row, min_col:max_col]
            output_valid = output_arr[min_row:max_row, min_col:max_col]
            
            target_valid = target_arr[min_row:max_row, min_col:max_col]
            
            if (output_valid == target_valid).all():

                counter+=1
              # Remove this break to print all test samples
print("Input:")
print(input_valid)
print("Output:")
print(output_valid)
print("Target:")
print(target_valid)
print("\n")
print(counter)

print(f'Test Loss: {test_loss/len(test_dataloader)}')
