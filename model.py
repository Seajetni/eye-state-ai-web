import torch
import torch.nn as nn

class EyeCNN(nn.Module):
    """
    Convolutional Neural Network for Open vs Closed Eyes Classification.
    Input shape: (batch_size, 3, 64, 64)
    Output shape: (batch_size, 2) -> [logits_closed, logits_open]
    """
    def __init__(self, num_classes=2):
        super(EyeCNN, self).__init__()
        
        # Block 1: 64x64 -> 32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 32x32 -> 16x16
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 16x16 -> 8x8
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        logits = self.classifier(x)
        return logits

def get_model(num_classes=2):
    return EyeCNN(num_classes=num_classes)

if __name__ == "__main__":
    model = get_model()
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    print("Model initialized successfully!")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
