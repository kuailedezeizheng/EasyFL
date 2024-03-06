import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import utils
import snntorch.functional as SF


def forward_pass(net, data, num_steps):
    spk_rec = []  # record spikes over time
    # reset/initialize hidden states for all LIF neurons in net
    utils.reset(net)

    for step in range(num_steps):  # loop over time
        spk_out, mem_out = net(data)  # one time step of the forward-pass
        spk_rec.append(spk_out)  # record spikes

    return torch.stack(spk_rec)


batch_size = 128
data_path = '../data/mnist'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(
    data_path,
    train=True,
    download=True,
    transform=transform)
mnist_test = datasets.MNIST(
    data_path,
    train=False,
    download=True,
    transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

beta = 0.9  # neuron decay rate
spike_grad = surrogate.fast_sigmoid()  # fast sigmoid surrogate gradient

#  Initialize Convolutional SNN
net = nn.Sequential(
    nn.Conv2d(
        1,
        8,
        5),
    nn.MaxPool2d(2),
    snn.Leaky(
        beta=beta,
        spike_grad=spike_grad,
        init_hidden=True),
    nn.Conv2d(
        8,
        16,
        5),
    nn.MaxPool2d(2),
    snn.Leaky(
        beta=beta,
        spike_grad=spike_grad,
        init_hidden=True),
    nn.Flatten(),
    nn.Linear(
        16 * 4 * 4,
        10),
    snn.Leaky(
        beta=beta,
        spike_grad=spike_grad,
        init_hidden=True,
        output=True)).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 1  # run for 1 epoch - each data sample is seen only once
num_steps = 25  # run for 25 time steps

loss_hist = []  # record loss over iterations
acc_hist = []  # record accuracy over iterations

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec = forward_pass(net, data, num_steps)  # forward-pass
        loss_val = loss_fn(spk_rec, targets)  # loss calculation
        optimizer.zero_grad()  # null gradients
        loss_val.backward()  # calculate gradients
        optimizer.step()  # update weights
        loss_hist.append(loss_val.item())  # store loss

        # print every 25 iterations
        if i % 25 == 0:
            print(
                f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")
