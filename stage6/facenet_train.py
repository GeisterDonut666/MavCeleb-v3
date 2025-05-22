from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training    # This line may show errors, but as long the imports run, they can be ignored
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import torch; print(torch.cuda.is_available())
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Setup hardware
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
n_workers = 0 if os.name == 'nt' else 8
print('Running on device: {}'.format(device))

# Training hyperparams
batch_size = 64
epochs = 2
max_epochs_without_improvement = 5

# Set dataset paths and model save path
train_dir = os.path.join('train_test_split_aug', 'train')
valid_dir = os.path.join('train_test_split_aug', 'valid')
model_path = os.path.join('models', 'facenet_finetuned_adam_6.pt')

# Create datasets and DataLoaders
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

train_data = datasets.ImageFolder(train_dir, transform=trans)
valid_data = datasets.ImageFolder(valid_dir, transform=trans)
print(train_data.class_to_idx)
print(valid_data.class_to_idx)

train_loader = DataLoader(
    train_data,
    num_workers=n_workers,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    valid_data,
    num_workers=n_workers,
    batch_size=batch_size,
    shuffle=True
)

# Training
resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=len(train_data.class_to_idx)
).to(device)

optimizer = torch.optim.Adam(resnet.parameters(), lr=0.00001)
#optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-6)
scheduler = MultiStepLR(optimizer, [5, 10])

loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
valid_loss, valid_metrics = training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

best_valid_loss = valid_loss
epochs_without_improvement = 0
train_losses = []
valid_losses = []

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    train_loss, _ = training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
    train_losses.append(train_loss)

    resnet.eval()
    valid_loss, _ = training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )
    valid_losses.append(valid_loss)

    if valid_loss <= best_valid_loss:
        torch.save(resnet, model_path)
        best_valid_loss = valid_loss
        epochs_without_improvement = 0
    else:
        if epochs_without_improvement >= max_epochs_without_improvement:
            break
        epochs_without_improvement += 1

writer.close()

# Plot accuracies
plt.figure(),
plt.plot(np.arange(len(train_losses)), train_losses),
plt.plot(np.arange(len(valid_losses)), valid_losses),
plt.legend(['training', 'validation']),
plt.xlabel('epoch'),
plt.ylabel('loss'),
plt.show()
