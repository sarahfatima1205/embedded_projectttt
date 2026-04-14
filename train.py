import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# 🔥 SMALL MODEL (IMPORTANT)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 16)   
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 10)     

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Save
torch.save(model.state_dict(), "weights/model.pth")

# Export first test sample as a C array for embedded testing
model.eval()
sample, label = test_dataset[0]

# Quantize using same scale as input (normalize was (0.5, 0.5) so range is [-1, 1])
# Map [-1,1] float → [-127,127] int8
s_input = 127.0
q_sample = (sample.view(-1) * s_input).round().clamp(-128, 127).numpy().astype(np.int8)

with open("test_sample.h", "w") as f:
    f.write("#ifndef TEST_SAMPLE_H\n#define TEST_SAMPLE_H\n\n")
    f.write(f"// Label: {label}\n")
    f.write(f"#define TEST_SAMPLE_LABEL {label}\n\n")
    f.write(f"const int8_t test_sample[784] = {{\n")
    for i in range(0, 784, 28):
        f.write("  " + ",".join(map(str, q_sample[i:i+28])) + ",\n")
    f.write("};\n\n#endif\n")

print(f"Test sample exported — label is {label}")

print("Training done + model saved")