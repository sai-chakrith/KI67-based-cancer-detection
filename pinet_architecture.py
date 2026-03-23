wheimport torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Ki67Dataset(Dataset):
    """
    Custom Dataset for loading the Ki67 images from the allData folder.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Assuming you will load labels/masks here. For now, returning dummy targets
        dummy_target = torch.zeros((1, 256, 256))
        
        return image, dummy_target

class ResBlock(nn.Module):
    """
    A Residual Block used in the ResUNet architecture.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        # A 1x1 conv to match dimensions if in_channels != out_channels or stride != 1
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class piNET(nn.Module):
    """
    piNET Framework: An Automated Proliferation Index Calculator based on U-Net & ResUNet.
    Designed to accept 256x256 images and extract features mimicking the piNET deep learning architecture.
    """
    def __init__(self, in_channels=3, out_classes=1): # Usually out_classes = 1 or 2 (Ki67 positive/negative)
        super(piNET, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = ResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ResBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ResBlock(512, 1024)
        
        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResBlock(1024, 512) # 1024 because of concatenation from enc4
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResBlock(128, 64)
        
        # Final output convolution
        self.final_conv = nn.Conv2d(64, out_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        u4 = self.upconv4(b)
        c4 = torch.cat((e4, u4), dim=1) # Skip connection
        d4 = self.dec4(c4)
        
        u3 = self.upconv3(d4)
        c3 = torch.cat((e3, u3), dim=1) # Skip connection
        d3 = self.dec3(c3)
        
        u2 = self.upconv2(d3)
        c2 = torch.cat((e2, u2), dim=1) # Skip connection
        d2 = self.dec2(c2)
        
        u1 = self.upconv1(d2)
        c1 = torch.cat((e1, u1), dim=1) # Skip connection
        d1 = self.dec1(c1)
        
        # Final prediction layer
        out = self.final_conv(d1)
        
        # Standard U-Net output usually pushed through Sigmoid for binary prediction (e.g. heatmap of nuclei)
        return torch.sigmoid(out)

if __name__ == "__main__":
    import torchvision.transforms as transforms
    
    # Setup data transformation to resize images to 256x256 and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Initialize the Dataset and DataLoader
    data_path = "./allData"
    dataset = Ki67Dataset(data_dir=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Loaded {len(dataset)} images from {data_path}")
    
    # Test the piNET architecture with actual data
    model = piNET(in_channels=3, out_classes=1)
    
    # Fetch a batch from the dataloader
    images, targets = next(iter(dataloader))
    
    output = model(images)
    print(f"Real Batch Input Shape: {images.shape}")
    print(f"Model Output Shape: {output.shape}")
    print("piNET architecture forward pass with loaded data successful!")
