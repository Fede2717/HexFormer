import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from lib.geoopt import ManifoldParameter
from lib.geoopt.optim import RiemannianAdamW, RiemannianSGD

from lib.utils.imagenet import ImageNet

from models.classifier import ViTClassifier

from lib.utils.scheduler import build_scheduler

from lib.utils.autoaug import CIFAR10Policy
from lib.utils.random_erasing import RandomErasing
from lib.utils.sampler import RASampler


def load_checkpoint(model, optimizer, lr_scheduler, args):
    """ Loads a checkpoint from file-system. """

    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint:
        if checkpoint['args'].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr

            if (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(args.optimizer, checkpoint['args'].optimizer))

    epoch = 0
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1

    return model, optimizer, lr_scheduler, epoch

def load_model_checkpoint(model, checkpoint_path):
    """ Loads a checkpoint from file-system. """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model

def select_model(img_dim, num_classes, args):
    """ Selects and sets up an available model and returns it. """

    enc_args = {
        'num_layers' : args.num_layers,
        'img_dim' : img_dim,
        'num_classes' : num_classes,
        'patch_size' : args.patch_size,
        'heads' : args.num_heads,
        'hidden_dim' : args.hidden_dim,
        'mlp_dim' : args.mlp_dim,
    }

    if (args.encoder_manifold=="lorentz") or (args.encoder_manifold=="poincare"):
        enc_args['learn_k'] = args.learn_k
        enc_args['k'] = args.encoder_k

    dec_args = {
        'embed_dim' : args.hidden_dim,
        'num_classes' : num_classes,
        'k' : args.decoder_k,
        'learn_k' : args.learn_k,
        'type' : 'mlr',
        'clip_r' : args.clip_features
    }

    model = ViTClassifier(
        enc_type=args.encoder_manifold,
        dec_type=args.decoder_manifold,
        enc_kwargs=enc_args,
        dec_kwargs=dec_args
    )

    return model

def select_optimizer(model, len_train_loader, args):
    """ Selects and sets up an available optimizer and returns it. """

    model_parameters = get_param_groups(model, args.lr, args.weight_decay)

    if args.optimizer == "RiemannianAdamW":
        optimizer = RiemannianAdamW(model_parameters, lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, stabilize=1)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model_parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model_parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model
      
    lr_scheduler = build_scheduler(args, optimizer, len_train_loader)

    return optimizer, lr_scheduler

def get_param_groups(model, lr_manifold, weight_decay_manifold):
    no_decay = ["scale"]
    k_params = [".k"]

    parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and not any(nd in n for nd in no_decay)
                and not isinstance(p, ManifoldParameter)
                and not any(nd in n for nd in k_params)
            ],
            "name": "1"
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and isinstance(p, ManifoldParameter)
            ],
            'lr': lr_manifold,
            "weight_decay": weight_decay_manifold,
            "name": "manifold"
        },
        {  # k parameters
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad
                and any(nd in n for nd in k_params)
            ],
            "weight_decay": weight_decay_manifold,
            "lr": 1e-1,
            "name": "k_group"
        }
    ]

    return parameters

def select_dataset(args, validation_split=False):
    """ Selects an available dataset and returns PyTorch dataloaders for training, validation and testing. """

    if args.dataset == 'CIFAR-10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)

        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=mean)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_set = datasets.CIFAR10('data', train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR10('data', train=False, download=False, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 10

    elif args.dataset == 'CIFAR-100':
        mean = (0.5070, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=mean)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_set = datasets.CIFAR100('data', train=True, download=True, transform=train_transform)
        if validation_split:
            train_set, val_set = torch.utils.data.random_split(train_set, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR100('data', train=False, download=False, transform=test_transform)

        img_dim = [3, 32, 32]
        num_classes = 100

    elif args.dataset == 'Tiny-ImageNet':
        root_dir = "/add/path/here/" 
        train_dir = root_dir + "train"
        val_dir = root_dir + "val"
        test_dir = root_dir + "val" # No labels for test were given, so treat validation as test

        mean = (0.4802, 0.4481, 0.3975)
        std = (0.2770, 0.2691, 0.2821)

        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=mean)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_set = datasets.ImageFolder(train_dir, train_transform)
        val_set = datasets.ImageFolder(val_dir, test_transform)
        test_set = datasets.ImageFolder(test_dir, test_transform)

        img_dim = [3, 64, 64]
        num_classes = 200

    elif args.dataset == 'ImageNet':
        root_dir = "classification/data/imagenet/"

        train_transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_set = ImageNet(root_dir, split='train', transform=train_transform)
        val_set = ImageNet(root_dir, split='val', transform=test_transform)
        test_set = ImageNet(root_dir, split='val', transform=test_transform)

        img_dim = [3, 224, 224]
        num_classes = 1000

    else:
        raise "Selected dataset '{}' not available.".format(args.dataset)
    
    # Dataloader
    train_loader = DataLoader(train_set,
        num_workers=4, 
        pin_memory=True, 
        batch_sampler=RASampler(len(train_set), 
            batch_size=args.batch_size, 
            repetitions=1,
            len_factor=3,
            shuffle=True, 
            drop_last=True
        )
    )
    test_loader = DataLoader(test_set, 
        batch_size=args.batch_size_test, 
        num_workers=4, 
        pin_memory=True, 
        shuffle=False
    ) 
    
    if validation_split:
        val_loader = DataLoader(val_set, 
            batch_size=args.batch_size_test, 
            num_workers=4, 
            pin_memory=True, 
            shuffle=False
        )
    else:
        val_loader = test_loader
        
    return train_loader, test_loader, val_loader, img_dim, num_classes
