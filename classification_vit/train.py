# -----------------------------------------------------
# Change working directory to parent 
import os
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import torch
from torch.nn import DataParallel
import configargparse
from tqdm import tqdm
import random
import numpy as np
import csv  # Import CSV module for saving the metrics
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import datetime

from utils.initialize import select_dataset, select_model, select_optimizer, load_checkpoint
from lib.utils.utils import AverageMeter, accuracy
from lib.utils.mix import cutmix_data, mixup_data, mixup_criterion
from lib.utils.losses import LabelSmoothingCrossEntropy
from haa_diagnostics import log_haa_epoch_metrics, log_haa_deep_diagnostics

os.environ['WANDB_DIR'] = '/media/hdd/usr/forner/wandb/'


def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='Image classification training', add_help=True)

    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str,
                        help="Path to config file.")

    # Output settings
    parser.add_argument('--exp_name', default="test", type=str,
                        help="Name of the experiment.")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Path for output files (relative to working directory).")

    # General settings
    parser.add_argument('--device', default="cuda:0",
                        type=lambda s: [str(item) for item in s.replace(' ', '').split(',')],
                        help="List of devices split by comma (e.g. cuda:0,cuda:1), can also be a single device or 'cpu')")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"],
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int,
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help="Path to model checkpoint (weights, optimizer, epoch).")
    parser.add_argument('--compile', action='store_true',
                        help="Compile model for faster inference (requires PyTorch 2).")
    parser.add_argument('--histogram', action='store_true',
                        help="Have a gradients histogram in tensorboard.")

    # General training parameters
    parser.add_argument('--num_epochs', default=200, type=int,
                        help="Number of training epochs.")
    parser.add_argument('--warmup', default=10, type=int,
                        help="Number of warmup epochs.")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="Training batch size.")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Training learning rate.")
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--optimizer', default="RiemannianAdamW", type=str,
                        choices=["RiemannianAdamW", "RiemannianSGD", "AdamW", "SGD"],
                        help="Optimizer for training.")

    # General validation/testing hyperparameters
    parser.add_argument('--batch_size_test', default=128, type=int,
                        help="Validation/Testing batch size.")

    # Transformer settings
    parser.add_argument('--patch_size', default=16, type=int,
                        help="Number of encoder layers in ViT.")
    # Add model size option
    parser.add_argument(
        "--model_size",
        choices=["tiny", "small", "base"],
        default="tiny",
        help="Choose model size: tiny, small, or base."
    )
    parser.add_argument('--num_layers', default=None, type=int,
                        help="Number of encoder layers in ViT.")
    parser.add_argument('--hidden_dim', default=None, type=int,
                        help="Dimensionality of hidden dimensionality")
    parser.add_argument('--mlp_dim', default=None, type=int,
                        help="Dimensionality of hidden dimensionality")
    parser.add_argument('--num_heads', default=None, type=int,
                        help="Number of heads in Transformer attention")
    parser.add_argument('--encoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz", "poincare"],
                        help="Select conv model encoder manifold.")
    parser.add_argument('--decoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz", "poincare"],
                        help="Select conv model decoder manifold.")

    # Hyperbolic geometry settings
    parser.add_argument('--learn_k', action='store_true',
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--encoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--decoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).")
    parser.add_argument('--clip_features', default=1.0, type=float,
                        help="Clipping parameter for hybrid HNNs proposed by Guo et al. (2022)")

    # Dataset settings
    parser.add_argument('--dataset', default='CIFAR-100', type=str,
                        choices=["CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "ImageNet"],
                        help="Select a dataset.")

    # HAA ablation mode
    parser.add_argument('--haa_mode', default='baseline', type=str,
                        choices=['baseline', 'terminal', 'full_uniform', 'continuous', 'terminal_aggressive'],
                        help="HAA injection mode: baseline=no HAA, terminal=last layer only, full_uniform=all layers, continuous=all layers with depth-proportional beta init, terminal_aggressive=last layer with high tau/low lambda.")
    parser.add_argument('--haa_tau_init', default=0.1, type=float,
        help="Initial tau value for HAA entailment weight. Default 0.1. Use 3.0 for terminal_aggressive.")
    parser.add_argument('--haa_lambda_init', default=1.0, type=float,
        help="Initial lambda value for HAA spatial weight. Default 1.0. Use 0.3 for terminal_aggressive.")
    parser.add_argument('--deep_diagnostics', action='store_true',
        help="Run extra val pass at epochs {1,5,10,20,final} for geometric diagnostics. "
             "Disable for RNG-clean training runs.")

    args = parser.parse_args()

    return args

def main(args):
    # Define presets
    model_configs = {
        "tiny":  dict(num_layers=9,  hidden_dim=192, mlp_dim=384,  num_heads=12),
        "small": dict(num_layers=12, hidden_dim=384, mlp_dim=768,  num_heads=6),
        "base":  dict(num_layers=12, hidden_dim=768, mlp_dim=3072, num_heads=12),
    }

    # Apply defaults if not explicitly set
    for key, value in model_configs[args.model_size].items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Map --haa_mode to the list of layer indices where HAA scoring is active
    _haa_mode_map = {
        'baseline':             [],
        'terminal':             [args.num_layers - 1],
        'full_uniform':         list(range(args.num_layers)),
        'continuous':           list(range(args.num_layers)),
        'terminal_aggressive':  [args.num_layers - 1],
    }
    args.active_haa_layers = _haa_mode_map[args.haa_mode]
    args.beta_proportional = (args.haa_mode == 'continuous')

    if args.haa_mode == 'terminal_aggressive':
        if args.haa_tau_init == 0.1:
            args.haa_tau_init = 3.0
        if args.haa_lambda_init == 1.0:
            args.haa_lambda_init = 0.3

    device = args.device[0]
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    print("Running experiment: " + args.exp_name)

    print("Arguments:")
    print(args)

    print("Loading dataset...")
    train_loader, val_loader, test_loader, img_dim, num_classes = select_dataset(args)

    print("Creating model...")
    model = select_model(img_dim, num_classes, args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, len(train_loader), args)
    criterion = LabelSmoothingCrossEntropy()

    start_epoch = 0
    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(model, optimizer, lr_scheduler, args)

    # model = DataParallel(model, device_ids=args.device)
    device_ids = [int(d.replace('cuda:', '')) for d in args.device]
    model = DataParallel(model, device_ids=device_ids)

    if args.compile:
        model = torch.compile(model)

    # Build a self-identifying experiment name that embeds the HAA mode
    if args.haa_mode != 'baseline':
        _suffix = args.haa_mode
        if args.haa_mode == 'terminal_aggressive':
            _suffix = (f"terminal_aggressive"
                       f"_tau{args.haa_tau_init}"
                       f"_lam{args.haa_lambda_init}")
        _run_exp_name = args.exp_name + "_haa_" + _suffix
    else:
        _run_exp_name = args.exp_name

    # Initialize TensorBoard writer
    # Set up TensorBoard logging directory with a unique name for each experiment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("/media/hdd/usr/forner/logs/", _run_exp_name + "_" + timestamp)
    writer = SummaryWriter(log_dir=log_dir)


    # Initialize lists to store the metrics
    train_losses, val_losses = [], []
    train_acc1s, val_acc1s = [], []
    train_acc5s, val_acc5s = [], []

    print("Training...")
    global_step = start_epoch * len(train_loader)

    best_acc = 0.0
    best_epoch = 0

    from lib.lorentz.blocks.transformer_blocks import LorentzMultiHeadAttention
    _base = model.module if hasattr(model, 'module') else model
    _haa_layers = sorted(
        [(m.layer_idx, m)
         for _, m in _base.named_modules()
         if isinstance(m, LorentzMultiHeadAttention) and m.use_haa],
        key=lambda x: x[0]
    )
    # Collect per-epoch HAA scalars — use last HAA layer only for CSV
    # (full per-layer data remains in TensorBoard)
    _haa_epoch_data = {
        'beta':             [],
        'tau':              [],
        'lambda':           [],
        'cone_sparsity':    [],
        'z_mean':           [],
        'frac_near_origin': [],
        'grad_norm_B':      [],
        'grad_norm_c_tilde':[],
    }

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        losses = AverageMeter("Loss", ":.4e")
        acc1 = AverageMeter("Acc@1", ":6.2f")
        acc5 = AverageMeter("Acc@5", ":6.2f")

        for i, (x, y) in tqdm(enumerate(train_loader)):
            # ------- Start iteration -------
            x = x.to(device)
            y = y.to(device)

            # Cutmix and Mixup
            r = np.random.rand(1)
            mix_prob = 0.5
            if r < mix_prob:
                switching_prob = np.random.rand(1)

                # Cutmix
                if switching_prob < 0.5:
                    slicing_idx, y_a, y_b, lam, sliced = cutmix_data(x, y)
                    x[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
                    output = model(x)
                    
                    loss =  mixup_criterion(criterion, output, y_a, y_b, lam)
                # Mixup
                else:
                    x, y_a, y_b, lam = mixup_data(x, y)
                    output = model(x)
                    
                    loss = mixup_criterion(criterion, output, y_a, y_b, lam) 
            else:
                output = model(x)
                loss = criterion(output, y) 

            optimizer.zero_grad()
            loss.backward()
            
            if args.histogram:
                if (i > 0 and i % 300 == 0) or i in [1, 5, 10, 25, 50, 75, 100, 150, 200]:
                    # Collect combined gradients
                    combined_gradients = []
                    for name, param in model.module.named_parameters():
                        if param.grad is not None:
                            combined_gradients.extend(param.grad.abs().add_(1e-9).log().view(-1).tolist())
                    
                    # Convert to tensor
                    combined_gradients_tensor = torch.tensor(combined_gradients, device=device)
                    
                    # Log close-up histogram for specific iterations in the list
                    if epoch == start_epoch:
                        if i in [1, 5, 10, 25, 50, 75, 100, 150, 200]:
                            writer.add_histogram('grad/combined_closeup', combined_gradients_tensor, global_step)
                    
                    # Log complete histogram for all iterations matching the condition
                    writer.add_histogram('grad/combined_complete', combined_gradients_tensor, global_step)


            optimizer.step()
            lr_scheduler.step()

            with torch.no_grad():
                top1, top5 = accuracy(output, y, topk=(1, 5))
                losses.update(loss.item())
                acc1.update(top1.item())
                acc5.update(top5.item())

            global_step += 1
            
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            # ------- End iteration -------
        # Log training metrics to TensorBoard
        writer.add_scalar('Loss/train', losses.avg, epoch)
        writer.add_scalar('Accuracy/train_top1', acc1.avg, epoch)
        writer.add_scalar('Accuracy/train_top5', acc5.avg, epoch)

        # Append the training metrics for this epoch
        train_losses.append(losses.avg)
        train_acc1s.append(acc1.avg)
        train_acc5s.append(acc5.avg)

        # ------- Start validation and logging -------
        with torch.no_grad():
            loss_val, acc1_val, acc5_val = evaluate(model, val_loader, criterion, device)

            print(
                "Epoch {}/{}: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}, Validation: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
                    epoch + 1, args.num_epochs, losses.avg, acc1.avg, acc5.avg, loss_val, acc1_val, acc5_val))

            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val', loss_val, epoch)
            writer.add_scalar('Accuracy/val_top1', acc1_val, epoch)
            writer.add_scalar('Accuracy/val_top5', acc5_val, epoch)

            # Log per-layer HAA telemetry (reads values stored during evaluate())
            log_haa_epoch_metrics(model, epoch, writer)

            # Collect HAA scalars from the last HAA layer for CSV
            if _haa_layers:
                _, _last_mha = _haa_layers[-1]
                _haa_epoch_data['beta'].append(_last_mha.haa_alpha)
                _haa_epoch_data['tau'].append(_last_mha.haa_tau)
                _haa_epoch_data['lambda'].append(_last_mha.haa_lambda)
                _haa_epoch_data['cone_sparsity'].append(_last_mha.haa_cone_sparsity)
                _haa_epoch_data['z_mean'].append(_last_mha.haa_mean_Z)
                _haa_epoch_data['frac_near_origin'].append(_last_mha.haa_frac_near_origin)
                _haa_epoch_data['grad_norm_B'].append(_last_mha._grad_norms.get('B', ''))
                _haa_epoch_data['grad_norm_c_tilde'].append(_last_mha._grad_norms.get('c_tilde', ''))

            # Append the validation metrics for this epoch
            val_losses.append(loss_val)
            val_acc1s.append(acc1_val)
            val_acc5s.append(acc5_val)

            # Testing for best model
            if acc1_val > best_acc:
                best_acc = acc1_val
                best_epoch = epoch + 1
                if args.output_dir is not None:
                    save_path = os.path.join(args.output_dir, f"best_{_run_exp_name}.pth")
                    torch.save({
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'epoch': epoch,
                        'args': args,
                    }, save_path)

        # Deep diagnostics at epochs 1, 5, 10, 20, and the final epoch
        if args.deep_diagnostics and ((epoch+1) in {1,5,10,20}
                                       or (epoch+1) == args.num_epochs):
            _K = model.module.enc_manifold.k.item()
            log_haa_deep_diagnostics(model, val_loader, device, epoch, _K, writer)
        # ------- End validation and logging -------

    print("-----------------\nTraining finished\n-----------------")
    print("Best epoch = {}, with Acc@1={:.4f}".format(best_epoch, best_acc))

    if args.output_dir is not None:
        save_path = os.path.join(args.output_dir, f"final_{_run_exp_name}.pth")
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epoch': epoch,
            'args': args,
        }, save_path)
        print("Model saved to " + save_path)

        # Save metrics to CSV
        metrics_file = os.path.join(args.output_dir, f"{_run_exp_name}_metrics.csv")
        with open(metrics_file, mode='w', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow([
                "Epoch",
                "Train Loss", "Val Loss",
                "Train Acc@1", "Val Acc@1",
                "Train Acc@5", "Val Acc@5",
                "haa_beta", "haa_tau", "haa_lambda",
                "haa_cone_sparsity", "haa_z_mean",
                "haa_frac_near_origin",
                "haa_grad_norm_B", "haa_grad_norm_c_tilde",
            ])
            for i in range(args.num_epochs):
                writer_csv.writerow([
                    i+1,
                    train_losses[i], val_losses[i],
                    train_acc1s[i], val_acc1s[i],
                    train_acc5s[i], val_acc5s[i],
                    _haa_epoch_data['beta'][i]             if i < len(_haa_epoch_data['beta']) else '',
                    _haa_epoch_data['tau'][i]              if i < len(_haa_epoch_data['tau']) else '',
                    _haa_epoch_data['lambda'][i]           if i < len(_haa_epoch_data['lambda']) else '',
                    _haa_epoch_data['cone_sparsity'][i]    if i < len(_haa_epoch_data['cone_sparsity']) else '',
                    _haa_epoch_data['z_mean'][i]           if i < len(_haa_epoch_data['z_mean']) else '',
                    _haa_epoch_data['frac_near_origin'][i] if i < len(_haa_epoch_data['frac_near_origin']) else '',
                    _haa_epoch_data['grad_norm_B'][i]      if i < len(_haa_epoch_data['grad_norm_B']) else '',
                    _haa_epoch_data['grad_norm_c_tilde'][i] if i < len(_haa_epoch_data['grad_norm_c_tilde']) else '',
                ])
        print(f"Metrics saved to {metrics_file}")

        # Save best accuracy and epoch to a text file
        best_metrics_file = os.path.join(args.output_dir, f"{_run_exp_name}_best_metrics.txt")
        with open(best_metrics_file, mode='w') as file:
            file.write(f"Best Epoch: {best_epoch}\n")
            file.write(f"Best Accuracy@1: {best_acc:.4f}\n\n")
            file.write(f"Arguments: {args}\n")
        print(f"Best metrics saved to {best_metrics_file}")
    else:
        print("Model and metrics not saved.")

    print("Testing final model...")
    loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

    print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
        loss_test, acc1_test, acc5_test))

    print("Testing best model...")
    if args.output_dir is not None:
        print("Loading best model...")
        save_path = os.path.join(args.output_dir, f"best_{_run_exp_name}.pth")
        checkpoint = torch.load(save_path, map_location=device)
        model.module.load_state_dict(checkpoint['model'], strict=True)

        loss_test, acc1_test, acc5_test = evaluate(model, test_loader, criterion, device)

        print("Results: Loss={:.4f}, Acc@1={:.4f}, Acc@5={:.4f}".format(
            loss_test, acc1_test, acc5_test))
    else:
        print("Best model not saved, because no output_dir given.")

    writer.close()


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """ Evaluates model performance """
    model.eval()
    model.to(device)

    losses = AverageMeter("Loss", ":.4e")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    acc5 = AverageMeter("Acc@5", ":6.2f")

    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)

        top1, top5 = accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item())
        acc1.update(top1.item(), x.shape[0])
        acc5.update(top5.item(), x.shape[0])

    return losses.avg, acc1.avg, acc5.avg

if __name__ == '__main__':
    args = getArguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + args.dtype

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            print("Create missing output directory...")
            os.mkdir(args.output_dir)

    main(args)