import argparse
from pprint import pprint

import numpy as np
import torch
from avalanche.benchmarks.scenarios import split_online_stream
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import as_multitask
from avalanche.training.supervised import (AGEM, DER, ER_ACE, GEM, MIR, GDumb,
                                           JointTraining, Naive)
from avalanche.training.storage_policy import ClassBalancedBuffer
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader

from helpers import load_dataset, set_seed
from torchvision.models import mobilenet_v3_small, mobilenet_v2, resnet18
from model import MLP, create_model, CNN, CNNNB, SlimResNet18
from mir import MIRPlugin
from replay import Replay
from replaynu import ReplayNU
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
from avalanche.training.plugins import ReplayPlugin
from nuder import NUDER
from gss import GSS_greedyPlugin
import hashlib
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import json


def fingerprint(x, y, t=None):
    """
    Produce a fast, deterministic hash string for a single sample.
    """
    # flatten tensor to bytes
    xb = x.cpu().numpy().tobytes()
    return hashlib.md5(xb).hexdigest()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        default="cifar10",
        choices=["cifar10", "cifar100", "img", "fmnist", "mnist", "nette"],
    )
    parser.add_argument("--model", default="cnn", choices=["resnet18", "mlp", "mobilenet", "cnn", "cnnnb", "resnet18_pretrained", "mobilenet_pretrained"])
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training",
    )
    parser.add_argument(
        "--batch_size_mem",
        type=int,
        default=32,
        help="memory batch size for training",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--trials", type=int, default=3, help="number of random seeds")
    parser.add_argument("--vectors", type=int, default=5, help="number of random seeds")
    parser.add_argument("--memory_size", type=int, default=1000, help="replay memory size")
    parser.add_argument("--subsample", type=int, default=50, help='MIR method parameter')
    parser.add_argument(
        "--strategy",
        default="er",
        choices=["er", "der", "gem", "agem", "ace", "mir", "nu", "nuder", "gss", "nugss"],
        help="Choose the replay strategy",
    )
    parser.add_argument("--online", type=bool, default=False)
    parser.add_argument("--remove_current", type=bool, default=False, help="Do not sample from the memory buffer the samples that belong to the classes that are in the current training batch.")
    parser.add_argument("--alpha", type=float, default=0.1, help='DER method parameter')
    parser.add_argument("--beta", type=float, default=0.5, help='DER method parameter')
    parser.add_argument("--ti", type=bool, default=False, help='task incremental')
    parser.add_argument("--mem_strength", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def test(model, loader, device, args):
    model.eval()
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for images, labels, task_id in loader:
            images = images.to(device)
            if args.ti:
                pred = model(images, task_id)
            else:
                pred = model(images)
            pred = torch.max(pred.data, 1)[1].cpu()
            correct += (pred == labels.data).sum().numpy()
            total += labels.size(0)
    model.train()
    return correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_model(args):
    if args.dataset in ['cifar10']:
        n_classes = 10
        n_tasks = 5
        shape = (3, 32, 32)
    elif args.dataset in ['img']:
        n_classes = 200
        n_tasks = 10
        shape = (3, 64, 64)
    elif args.dataset in ['cub200']:
        n_classes = 200
        n_tasks = 5
        shape = (3, 224, 224)
    elif args.dataset in ['nette']:
        n_classes = 10
        n_tasks = 5
        shape = (3, 224, 224)
    elif 'mnist' in args.dataset:
        n_classes = 10
        n_tasks = 5
        shape = (1, 28, 28)
    else:
        n_classes = 100
        n_tasks = 5
        shape = (3, 32, 32)

    if args.model == "mlp":
        model = MLP(input_size=28 * 28 * shape[0], output_size=n_classes)
    elif args.model == 'cnn':
        model = CNN(n_classes=n_classes)
    elif args.model == 'cnnnb':
        model = CNNNB(n_classes=n_classes)
    elif args.model == 'resnet18':
        model = resnet18(pretrained=False)
        if shape[1] < 224:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, n_classes)    
    elif args.model == 'resnet18_pretrained':
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif args.model == 'mobilenet':
        model = mobilenet_v3_small(pretrained=False)
        if shape[1] < 224:
            model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=n_classes),
        )
    elif args.model == 'mobilenet_pretrained':
        model = mobilenet_v3_small(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=n_classes),
        )

    else:
        raise ValueError(f"Unknown model type: {args.model}")


    print('Params count:', count_parameters(model))
    model.eval()
    #input = torch.randn(1, 3, 32, 32)  # Adjust to your input shape
    #flops = FlopCountAnalysis(model, input)
    #print(f"FLOPs: {flops.total()}")  # total FLOPs
    #print(flops.by_operator())        # per operator
    if args.ti:
        try:
            model = as_multitask(model, "classifier")
        except:
            model = as_multitask(model, "linear")

    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer, n_tasks


def prob_vectors(scenario, vectors, seed):
    # Define the path to the JSON file
    file_path = '/home/ipipan/Desktop/BMVC/.guild/runs/88b9df741bc84b738a41e55646f4e873/vectors.json'

    # Load the JSON file into a Python list of dictionaries
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    res = [data[6]]    
    return res


def main(args):
    if args.batch_size_mem == 0:
        args.batch_size_mem = args.memory_size
    if args.batch_size_mem < 0:
        args.batch_size_mem = args.batch_size
    pprint(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    totals = []
    accs = []
    if 'nu' in args.strategy:
        vectors = args.vectors
    else:
        vectors = 1
        
    scenario = load_dataset(args, 0)
    trial_custom_probs = prob_vectors(scenario, vectors, args.seed)

    v_accs = []
    for v in range(vectors):
        accs = []
        for iteration in range(args.trials):
            print("iteration:", iteration)
            print("vector:", v)
            set_seed(iteration)
            
            scenario = load_dataset(args, iteration)
            #trial_custom_probs = prob_vectors(scenario, args.vectors)
            model, optimizer, n_tasks = init_model(args)
    
            storage_policy = ClassBalancedBuffer(args.memory_size, adaptive_size=True)
            eval_plugin = EvaluationPlugin(
                         accuracy_metrics(minibatch=False, epoch=False, epoch_running=False, 
                         experience=True, stream=True))
            
            if args.strategy == "der":
                cl_strategy = DER(
                    model=model,
                    optimizer=optimizer,
                    #plugins=[],
                    device=device,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    mem_size=args.memory_size,
                    batch_size_mem=args.batch_size_mem,
                    train_epochs=args.epochs,
                    alpha=args.alpha,
                    beta=args.beta,
                    evaluator=eval_plugin,
                )                
            elif args.strategy == "nuder":
                cl_strategy = NUDER(
                    model=model,
                    optimizer=optimizer,
                    plugins=[],
                    device=device,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    mem_size=args.memory_size,
                    batch_size_mem=args.batch_size_mem,
                    train_epochs=args.epochs,
                    alpha=args.alpha,
                    beta=args.beta,
                    evaluator=eval_plugin,
                    remove_current=args.remove_current,
                    sample_weight_table=trial_custom_probs[v],
                )
            elif args.strategy == "ace":
                cl_strategy = ER_ACE(
                    model=model,
                    optimizer=optimizer,
                    plugins=[],
                    device=device,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    mem_size=args.memory_size,
                    train_epochs=args.epochs,
                    evaluator=eval_plugin,
                )
            elif args.strategy == "mir":
                if args.batch_size_mem < 50:
                    subsample = 50
                else:
                    subsample = args.memory_size
                    
                cl_strategy3 = ReplayNU(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size_mem=args.batch_size_mem,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    train_epochs=args.epochs,
                    mem_size=args.memory_size,
                    evaluator=eval_plugin,
                    remove_current=args.remove_current,
                    ti=args.ti,
                    sample_weight_table=[1]*args.memory_size,
                    plugins=[MIRPlugin(mem_size=args.memory_size, subsample=50, batch_size_mem=args.batch_size_mem)],
                )
                
                cl_strategy = Replay(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size_mem=args.batch_size_mem,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    train_epochs=args.epochs,
                    mem_size=args.memory_size,
                    evaluator=eval_plugin,
                    remove_current=args.remove_current,
                    ti=args.ti,
                    plugins=[MIRPlugin(mem_size=args.memory_size, subsample=subsample, batch_size_mem=args.batch_size_mem)],
                )                
                    
                cl_strategy2 = MIR(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size_mem=args.batch_size_mem,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    train_epochs=args.epochs,
                    mem_size=args.memory_size,
                    evaluator=eval_plugin,
                    subsample=subsample,
                    plugins=[ReplayPlugin(mem_size=args.memory_size)],
                    criterion=nn.CrossEntropyLoss()
                )
            elif args.strategy == "nu":
                cl_strategy = ReplayNU(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size_mem=args.batch_size_mem,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    train_epochs=args.epochs,
                    mem_size=args.memory_size,
                    evaluator=eval_plugin,
                    remove_current=args.remove_current,
                    ti=args.ti,
                    sample_weight_table=trial_custom_probs[v],
                )
            elif args.strategy == "gss":
                input_size = [3,32,32]
                if args.dataset == 'nette':
                    input_size = [3,244,244]
                cl_strategy = Replay(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size_mem=args.batch_size_mem,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    train_epochs=args.epochs,
                    mem_size=args.memory_size,
                    evaluator=eval_plugin,
                    remove_current=args.remove_current,
                    ti=args.ti,
                    plugins=[GSS_greedyPlugin(mem_size=args.memory_size, input_size=input_size, mem_strength=args.mem_strength)],
                )
            elif args.strategy == "nugss":
                input_size = [3,32,32]
                if args.dataset == 'nette':
                    input_size = [3,244,244]
                cl_strategy = ReplayNU(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size_mem=args.batch_size_mem,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    train_epochs=args.epochs,
                    mem_size=args.memory_size,
                    evaluator=eval_plugin,
                    remove_current=args.remove_current,
                    ti=args.ti,
                    sample_weight_table=trial_custom_probs[v],
                    plugins=[GSS_greedyPlugin(mem_size=args.memory_size, input_size=input_size, mem_strength=args.mem_strength)],
                )
            else:
                cl_strategy2 = ReplayNU(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size_mem=args.batch_size_mem,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    train_epochs=args.epochs,
                    mem_size=args.memory_size,
                    evaluator=eval_plugin,
                    remove_current=args.remove_current,
                    ti=args.ti,
                    sample_weight_table=[1]*args.memory_size,
                )
                cl_strategy = Replay(
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size_mem=args.batch_size_mem,
                    train_mb_size=args.batch_size,
                    eval_mb_size=64,
                    train_epochs=args.epochs,
                    mem_size=args.memory_size,
                    evaluator=eval_plugin,
                    remove_current=args.remove_current,
                    ti=args.ti,
                )
    
            if args.online:
                train_stream = split_online_stream(
                    original_stream=scenario.train_stream,
                    experience_size=args.batch_size,
                    access_task_boundaries=args.ti,
                )
                for task, experience in tqdm(enumerate(train_stream)):
                    cl_strategy.train(
                        experience,
                        eval_streams=[],
                        num_workers=0,
                        drop_last=True,
                    )
        
                task_accs = []
                for i in range(n_tasks):
                    test_loader = DataLoader(
                        scenario.test_stream[i].dataset,
                        batch_size=256,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=1,
                    )
                    test_acc = test(cl_strategy.model, test_loader, device, args)
                    task_accs.append(test_acc)
                    print('Task ' + str(i) + ':', test_acc)
                print("task_accs:", task_accs, np.mean(task_accs))
            else:
                train_stream = scenario.train_stream
    
                for task, experience in enumerate(train_stream):
                    cl_strategy.train(
                        experience,
                        eval_streams=[],
                        num_workers=0,
                        drop_last=True,
                    )
        
                    task_accs = []
                    for i in range(task+1):
                        test_loader = DataLoader(
                            scenario.test_stream[i].dataset,
                            batch_size=32,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=1,
                        )
                        task_accs.append(test(cl_strategy.model, test_loader, device, args))
                    print("task_accs:", task_accs, np.mean(task_accs))
            accs.append(np.mean(task_accs))
            print("accs_"+str(v)+":", accs)
            print("accuracy_"+str(v)+":", np.mean(accs))
            print("std_"+str(v)+":", np.std(accs))
        v_accs.append(np.mean(accs))
    print(v_accs)
    print('v_accs_mean:', np.mean(v_accs))
    print('v_accs_std:', np.std(v_accs))
    print('v_accs_max:', np.max(v_accs))
    print('v_accs_min:', np.min(v_accs))
        
    return np.mean(accs)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
