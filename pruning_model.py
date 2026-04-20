import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# DATASET
# -----------------------------
def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    print("[STEP] Loading CIFAR-10...")

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainset, range(10000))

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testset = torch.utils.data.Subset(testset, range(1000))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# -----------------------------
# PRUNABLE LAYER
# -----------------------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # IMPORTANT: start gates HIGH (avoid instant pruning)
        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features) * 2.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return torch.nn.functional.linear(x, pruned_weights, self.bias)

# -----------------------------
# MODEL
# -----------------------------
class PruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            PrunableLinear(3072, 256),
            nn.ReLU(),
            PrunableLinear(256, 128),
            nn.ReLU(),
            PrunableLinear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

    def sparsity_loss(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                loss += torch.sum(torch.sigmoid(m.gate_scores))
        return loss

# -----------------------------
# TRAINING
# -----------------------------
def train_model(model, trainloader, lmbda, epochs=15):

    weight_params = [p for n, p in model.named_parameters() if "gate" not in n]
    gate_params = [p for n, p in model.named_parameters() if "gate" in n]

    optimizer = optim.Adam([
        {'params': weight_params, 'lr': 0.001},
        {'params': gate_params, 'lr': 0.01}
    ])

    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # scaled sparsity (VERY IMPORTANT FIX)
            loss = criterion(model(x), y) + lmbda * 0.1 * model.sparsity_loss()

            loss.backward()
            optimizer.step()

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(model, testloader):
    model.eval()

    correct, total = 0, 0
    total_weights, pruned = 0, 0
    all_gates = []

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            _, pred = torch.max(out, 1)

            total += y.size(0)
            correct += (pred == y).sum().item()

    acc = 100 * correct / total

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = torch.sigmoid(m.gate_scores).detach().cpu().numpy()
            total_weights += g.size
            pruned += np.sum(g < 0.05)
            all_gates.extend(g.flatten())

    sparsity = 100 * pruned / total_weights

    return acc, sparsity, all_gates

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    trainloader, testloader = get_dataloaders()

    lambdas = [0.0001, 0.001, 0.01]

    results = []
    gate_data = {}

    for lmbda in lambdas:
        print(f"\n--- Training λ = {lmbda} ---")

        model = PruningNet().to(device)

        train_model(model, trainloader, lmbda)
        acc, sparsity, gates = evaluate(model, testloader)

        results.append((lmbda, acc, sparsity))
        gate_data[lmbda] = gates

        print(f"Accuracy: {acc:.2f}%")
        print(f"Sparsity: {sparsity:.2f}%")

    # -----------------------------
    # TABLE
    # -----------------------------
    print("\n==============================")
    print("Lambda | Accuracy | Sparsity")
    print("------------------------------")
    for r in results:
        print(f"{r[0]} | {r[1]:.2f}% | {r[2]:.2f}%")
    print("==============================")

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure(figsize=(15, 5))

    for i, lmbda in enumerate(lambdas):
        plt.subplot(1, 3, i+1)
        plt.hist(gate_data[lmbda], bins=50)
        plt.title(f"λ = {lmbda}")
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig("gate_distributions.png")

    print("\n✅ Plot saved as gate_distributions.png")