import os
import argparse
import time
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.models import mobilenet_v3_small
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from pprint import pprint

# --------------------------------------------------
# 0. IndexedDataset Wrapper
# --------------------------------------------------
class IndexedDataset(Dataset):
    """
    Wraps a dataset to return (data, label, index) on __getitem__.
    This is useful for knowing the original index of each sample.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx
    
    def __len__(self):
        return len(self.dataset)

# --------------------------------------------------
# 1. Simple CNN Model
# --------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_features=3136, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --------------------------------------------------
# 2. Seeding Functions
# --------------------------------------------------
def seed_everything(seed=42):
    """Seeds all randomness including numpy (for overall reproducibility)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_torch_and_python(seed=42):
    """
    Seeds only PyTorch and Python's random module.
    This is used at the beginning of each trial to ensure that the
    training order and memory updates are identical across trials.
    (NumPy is left unseeded here so that custom probability vectors vary.)
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------
# 3. Create Tasks for Continual Learning (Class-Incremental)
# --------------------------------------------------
def create_class_incremental_tasks(dataset_name, n_tasks=5, download=True):
    if dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        raw_train_set = torchvision.datasets.MNIST(root='~/data', train=True, download=download, transform=transform)
        raw_test_set  = torchvision.datasets.MNIST(root='~/data', train=False, download=download, transform=transform)
        input_channels = 1
        num_classes = 10
        num_features = 3136
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        raw_train_set = torchvision.datasets.CIFAR10(root='~/data', train=True, download=download, transform=transform)
        raw_test_set  = torchvision.datasets.CIFAR10(root='~/data', train=False, download=download, transform=transform)
        input_channels = 3
        num_classes = 10
        num_features = 4096
    elif dataset_name.lower() == 'img':
        transform = transforms.Compose([
            transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            raw_train_set = torchvision.datasets.Imagenette('~/data', split='train', download=True, size='full', transform=transform)
            raw_test_set = torchvision.datasets.Imagenette('~/data', split='val', download=True, size='full', transform=transform)
        except:
            raw_train_set = torchvision.datasets.Imagenette('~/data', split='train', download=False, size='full', transform=transform)
            raw_test_set = torchvision.datasets.Imagenette('~/data', split='val', download=False, size='full', transform=transform)
        input_channels = 3
        num_classes = 10
        num_features = 4096
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    classes_per_task = num_classes // n_tasks
    assert num_classes % n_tasks == 0, "n_tasks should evenly divide the number of classes."

    tasks_train = []
    tasks_test = []
    indexed_train_set = IndexedDataset(raw_train_set)
    indexed_test_set = IndexedDataset(raw_test_set)
    for task_id in range(n_tasks):
        class_start = task_id * classes_per_task
        class_end = class_start + classes_per_task
        classes_in_task = list(range(class_start, class_end))

        try:
            train_indices = [i for i, y in enumerate(raw_train_set.targets) if y in classes_in_task]
            test_indices = [i for i, y in enumerate(raw_test_set.targets) if y in classes_in_task]
        except:
            train_indices = [i for i, y in enumerate(np.array(raw_train_set._samples)[:,1].astype("int")) if y in classes_in_task]
            test_indices = [i for i, y in enumerate(np.array(raw_test_set._samples)[:,1].astype("int")) if y in classes_in_task]

        tasks_train.append(Subset(indexed_train_set, train_indices))
        tasks_test.append(Subset(indexed_test_set, test_indices))

    return tasks_train, tasks_test, input_channels, num_classes, raw_train_set, num_features

# --------------------------------------------------
# 4. Reservoir Sampling Helper
# --------------------------------------------------
def reservoir_update(buffer, sample, counter, max_size):
    """
    Update a reservoir (buffer) with a new sample using standard reservoir sampling.
    'counter' is the total number of samples seen so far.
    """
    counter += 1
    if len(buffer) < max_size:
        buffer.append(sample)
    else:
        if random.random() < (max_size / counter):
            replace_idx = random.randint(0, max_size - 1)
            buffer[replace_idx] = sample
    return counter

# --------------------------------------------------
# 5. Sampling from Memory (Replay)
# --------------------------------------------------
def sample_from_memory(memory_buffer, replay_batch_size, custom_probs, device):
    """
    Sample replay_batch_size samples from memory_buffer.
    If custom_probs is provided, use it to define the probability of sampling each sample.
    Because the memory buffer grows over time, only the first len(memory_buffer) elements
    of custom_probs are used (and renormalized). If custom_probs is None, uniform sampling is used.
    """
    if replay_batch_size <= 0 or len(memory_buffer) == 0:
        return torch.Tensor([]).to(device), torch.LongTensor([]).to(device)
    
    if custom_probs is not None:
        # Use the custom probabilities for the currently stored samples.
        probs = custom_probs[:len(memory_buffer)]
        probs = probs / np.sum(probs)
    else:
        probs = np.ones(len(memory_buffer)) / len(memory_buffer)
        
    chosen_indices = np.random.choice(len(memory_buffer), size=replay_batch_size,
                                      replace=(replay_batch_size > len(memory_buffer)),
                                      p=probs)
    
    sampled_imgs = []
    sampled_labels = []
    for idx in chosen_indices:
        img, label, _ = memory_buffer[idx]
        sampled_imgs.append(img.unsqueeze(0))
        sampled_labels.append(label)
    
    sampled_imgs = torch.cat(sampled_imgs, dim=0).to(device)
    sampled_labels = torch.LongTensor(sampled_labels).to(device)
    return sampled_imgs, sampled_labels

# --------------------------------------------------
# 6. Evaluation Function
# --------------------------------------------------
def evaluate_accuracy(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, *_ in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

# --------------------------------------------------
# 7. Training + Replay Loop for a Single Trial
# --------------------------------------------------
def train_and_evaluate_trial(args, device, master_seed, trial_custom_probs):
    """
    Runs one full training experiment (across all tasks) using a replay sampling strategy defined
    by trial_custom_probs. The memory buffer is updated during training in an identical manner across trials
    because we reset the PyTorch and Python RNGs to master_seed at the start.
    If trial_custom_probs is None, uniform (i.i.d.) replay is used (baseline).
    """
    # Reset PyTorch and Python randomness (but not NumPy, so that custom probability vectors vary)
    seed_torch_and_python(master_seed)
    
    # Create tasks
    tasks_train, tasks_test, input_channels, num_classes, raw_train, num_features = create_class_incremental_tasks(
        dataset_name=args.dataset, n_tasks=args.n_tasks
    )
    
    if args.dataset == 'IMG':
        model = mobilenet_v3_small(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=10, bias=True),
          )
        model = model.to(device)
    else:
        model = SimpleCNN(input_channels=input_channels, num_features=num_features, num_classes=num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Use a single memory buffer (updated identically across trials)
    memory_buffer = []
    memory_counter = 0

    best_acc_each_task = [0.0 for _ in range(args.n_tasks)]
    final_acc_each_task = [0.0 for _ in range(args.n_tasks)]
    total_time = 0.0

    for task_id in range(args.n_tasks):
        print(f"\n=== Training on Task {task_id} ===")
        # For deterministic mini-batch ordering, use a generator with master_seed.
        g = torch.Generator()
        g.manual_seed(master_seed)
        task_loader = DataLoader(tasks_train[task_id],
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=False,
                                 generator=g)
        model.train()
        for epoch in range(args.epochs_per_task):
            start_time = time.time()
            for images, labels, indices in tqdm(task_loader, desc=f"Task {task_id} Epoch {epoch+1}"):
                # Save incoming training samples for updating the memory buffer later.
                new_images = images.clone()
                new_labels = labels.clone()
                new_indices = indices.clone()
                
                # Replay step: sample from the current memory buffer.
                current_memory = memory_buffer
                if len(current_memory) > 0 and args.replay_batch_size > 0:
                    replay_images, replay_labels = sample_from_memory(
                        current_memory,
                        args.replay_batch_size,
                        trial_custom_probs,
                        device
                    )
                    images = torch.cat([images.to(device), replay_images], dim=0)
                    labels = torch.cat([labels.to(device), replay_labels], dim=0)
                else:
                    images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Update the memory buffer with each incoming sample.
                for i in range(new_images.size(0)):
                    sample_img = new_images[i].cpu()
                    sample_label = new_labels[i].item()
                    memory_counter = reservoir_update(memory_buffer, (sample_img, sample_label, False),
                                                      memory_counter, args.memory_size)
            end_time = time.time()
            epoch_time = end_time - start_time
            total_time += epoch_time

            acc_current = evaluate_accuracy(model, tasks_test[task_id], device)
            if acc_current > best_acc_each_task[task_id]:
                best_acc_each_task[task_id] = acc_current
            
            print(f"  Epoch {epoch+1}/{args.epochs_per_task}, Task {task_id}, Acc = {acc_current:.2f}%, Time = {epoch_time:.2f}s")
    
    # Final evaluation on all tasks.
    for task_id in range(args.n_tasks):
        final_acc = evaluate_accuracy(model, tasks_test[task_id], device)
        final_acc_each_task[task_id] = final_acc
        print(f"Final accuracy on Task {task_id} = {final_acc:.2f}%")
    
    forgetting_vals = [best_acc_each_task[t] - final_acc_each_task[t] for t in range(args.n_tasks)]
    avg_forgetting = np.mean(forgetting_vals)
    avg_final_acc = np.mean(final_acc_each_task)

    print("\n==== SUMMARY for this Trial ====")
    for t in range(args.n_tasks):
        print(f"Task {t}: Best Acc = {best_acc_each_task[t]:.2f}%, Final Acc = {final_acc_each_task[t]:.2f}%, Forgetting = {forgetting_vals[t]:.2f}%")
    print(f"Avg Final Accuracy: {avg_final_acc:.2f}%")
    print(f"Avg Forgetting: {avg_forgetting:.2f}%")
    print(f"Total training time: {total_time:.2f} seconds")
    
    return {
        "final_acc_each_task": final_acc_each_task,
        "best_acc_each_task": best_acc_each_task,
        "forgetting_vals": forgetting_vals,
        "avg_forgetting": avg_forgetting,
        "avg_final_acc": avg_final_acc,
        "total_time": total_time
    }

# --------------------------------------------------
# 8. Main + Argparse (100 Trials per Seed)
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", 
                        choices=["MNIST", "CIFAR10", "IMG"], help="Which dataset to use.")
    parser.add_argument("--n_tasks", type=int, default=5, 
                        help="Number of tasks (e.g., 5 for class-incremental).")
    parser.add_argument("--memory_size", type=int, default=500, 
                        help="Maximum size of the replay memory.")
    parser.add_argument("--replay_batch_size", type=int, default=32,
                        help="Number of samples to replay each batch.")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training on the current task.")
    parser.add_argument("--epochs_per_task", type=int, default=1, 
                        help="Number of epochs per task.")
    parser.add_argument("--lr", type=float, default=0.01, 
                        help="Learning rate for SGD.")
    parser.add_argument("--n_seeds", type=int, default=5, 
                        help="Number of random seeds to run.")
    parser.add_argument("--n_trials", type=int, default=50, 
                        help="Number of replay probability distributions to try per seed.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pprint(vars(args))
    print("Using device:", device)

    overall_results = {}

    # Seed NumPy once for reproducibility of custom probability generation.
    seed_everything(123)
    
    for seed in range(args.n_seeds):
        print("="*80)
        print(f"Starting experiments for seed {seed}")
        # master_seed fixes the training order and memory updates within a seed.
        master_seed = 1000 + seed
        seed_trial_results = []
        
        for trial in range(args.n_trials):
            if trial == 0:
                # Baseline: uniform (i.i.d.) replay.
                trial_custom_probs = None
                print(f"\n--- Seed {seed}, Trial {trial} (Baseline: uniform sampling) ---")
            else:
                # Generate a custom probability vector of length args.memory_size.
                trial_custom_probs = np.random.rand(args.memory_size)
                print(f"\n--- Seed {seed}, Trial {trial} (Custom sampling) ---")
            results = train_and_evaluate_trial(args, device, master_seed, trial_custom_probs)
            seed_trial_results.append(results['avg_final_acc'])
        
        baseline_acc = seed_trial_results[0]
        best_trial_acc = np.max(seed_trial_results)
        mean_trial_acc = np.mean(seed_trial_results)
        print(f"\nSeed {seed} Summary:")
        print(f"  Baseline (Trial 0) Accuracy: {baseline_acc:.2f}%")
        print(f"  Best Trial Accuracy: {best_trial_acc:.2f}%")
        print(f"  Mean Accuracy over Trials: {mean_trial_acc:.2f}%")
        overall_results[seed] = {
            "baseline": baseline_acc,
            "best": best_trial_acc,
            "mean": mean_trial_acc
        }
    
    # Compute overall statistics across seeds.
    baseline_accs = [overall_results[s]["baseline"] for s in overall_results]
    best_accs = [overall_results[s]["best"] for s in overall_results]
    mean_accs = [overall_results[s]["mean"] for s in overall_results]

    print("\n==== OVERALL RESULTS ACROSS SEEDS ====")
    print(f"Average Baseline Accuracy: {np.mean(baseline_accs):.2f}% ± {np.std(baseline_accs):.2f}%")
    print(f"Average Best Trial Accuracy: {np.mean(best_accs):.2f}% ± {np.std(best_accs):.2f}%")
    print(f"Average Mean Trials Accuracy: {np.mean(mean_accs):.2f}% ± {np.std(mean_accs):.2f}%")

if __name__ == "__main__":
    main()
