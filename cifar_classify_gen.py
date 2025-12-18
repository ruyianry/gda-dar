import os
import logging
import pickle
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset, ConcatDataset
from omegaconf import OmegaConf

import dnnlib
from dataset.ConcatIdentifiableDataset import ConcatIdentifiableDataset
from dnnlib.util import is_url
from loss.weighted_ce import WeightedCrossEntropyLoss
from sida_training.sida_networks import SongUNet_EncoderDecoder
from utils import load_pickle
from generate_new_images import generate_new_images_edm2, generate_new_images_edm
from utils.parse_dataset import initialize_dataset
from utils.parse_model import initialize_model
from torch_utils import distributed as dist

LOG_FORMAT = "%(asctime)-15s - %(module)-20s - %(levelname)-7s | %(message)s"
LOG_LEVEL = os.getenv("CIFAR_CLAS_GEN", "INFO")
logging.basicConfig(format=LOG_FORMAT, level=logging.getLevelName(LOG_LEVEL))

LOGGER = logging.getLogger(name='__main__')

@torch.no_grad()
def calculate_batch_alpha(D, synthetic_images, synthetic_labels, augment_pipe=None, num_classes=10):
    P_mean = -1.2
    P_std = 1.2

    # convert synthetic_labels to one-hot
    synthetic_labels = torch.nn.functional.one_hot(synthetic_labels, num_classes=num_classes).float()

    rnd_normal = torch.randn([synthetic_images.shape[0], 1, 1, 1], device=synthetic_images.device)
    sigma = (rnd_normal * P_std + P_mean).exp()

    if augment_pipe is not None:
        y_augmented, augment_labels = augment_pipe(synthetic_images)
    else:
        y_augmented = synthetic_images
        augment_labels = torch.zeros(synthetic_images.shape[0], 9).to(synthetic_images.device)

    n = torch.randn_like(y_augmented) * sigma  # noise [fake_batch]
    _, logit_fake = D(y_augmented + n, sigma, synthetic_labels,
                                    augment_labels=augment_labels,
                                    return_flag='encoder_decoder')
    logit_fake = torch.sigmoid(logit_fake)
    # todo: turning logit to probability (check the shape, is it already per sample or i need to reduce from the WHC)
    # in optimized D, the logit_fake are close to 1 for real samples and close to 0 for fake samples
    # give a per-sample, mean reduction -> can try per-sample alpha vs batch-wise alpha

    # remember, alpha needs to be capped at 1.0 (we can't allow synthetic samples to have more weight than real samples)

    # logit is [B, 1, patch_logits, patch_logits]
    aggregation_option = 'per_sample'
    if aggregation_option == 'per_sample':
        # for each sample, calculate the individual alpha by averaging the 8x8 patch logits
        # then clip the alpha to 1.0
        r = torch.mean(logit_fake, dim=[2, 3]).squeeze()
        # convert to weight alpha by (1 - r) / r
        alpha = (1 - r) / r # [B]
        alpha = torch.clamp(alpha, 0, 1)
    elif aggregation_option == 'per_batch':
        # for each batch, calculate the alpha by averaging the 8x8 patch logits
        # then clip the alpha to 1.0
        r = torch.mean(logit_fake).squeeze()
        # convert to weight alpha by (1 - r) / r
        alpha = (1 - r) / r
        alpha = torch.clamp(alpha, 0, 1)
        # make alpha [B]
        alpha = alpha.repeat(synthetic_images.shape[0])
    else:
        raise ValueError(f"Invalid aggregation option {aggregation_option}")

    return alpha



def train_one_epoch(model, criterion, optimizer, data_loader, epoch, D, D_augment, device='cpu'):
    LOGGER.info(f"Training epoch {epoch}")
    model.train()

    train_loss = .0
    correct, total = 0, 0
    for i, ((inputs, targets), dataset_idx) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        dataset_idx = dataset_idx.to(device)

        # select all non-zero dataset indices (i.e. synthetic samples) for D
        synthetic_mask = dataset_idx != 0
        synthetic_images = inputs[synthetic_mask]
        synthetic_labels = targets[synthetic_mask]

        # calculate alpha for the synthetic samples
        alpha = calculate_batch_alpha(D, synthetic_images, synthetic_labels, D_augment) # [size_of_synthetic_samples]

        # calculate the weight for the all samples. the weight for synthetic samples are calculated using alpha while the weight for real samples are 1
        weight = torch.ones_like(targets, dtype=torch.float32)
        weight[synthetic_mask] = alpha
        weight = weight.to(device)


        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, weight)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    current_lr = optimizer.param_groups[0]['lr']
    LOGGER.info(
        f"Epoch {epoch} | Loss {train_loss / len(data_loader)} | Accuracy {100. * correct / total} "
        f"| lr {current_lr}")


def evaluate(model, criterion, data_loader, epoch, device='cpu'):
    model.eval()
    loss = .0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    LOGGER.info(
        f"Epoch {epoch} | Test loss {loss / len(data_loader)} | Test accuracy {100. * correct / total}")
    return 100. * correct / total


def classify(model, generator, train_dataset, test_dataset, config,
             ratio, generator_param=None, D=None, D_augment_pipe=None,
             end_epoch=-1, start_epoch=0):
    if end_epoch == -1:
        end_epoch = config.training_params.epochs

    device = config.device
    model = model.to(device)
    generator = generator.to(device)
    D = D.to(device)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    else:
        LOGGER.warning("CUDA is not available. Training on CPU")

    # criterion = nn.CrossEntropyLoss(label_smoothing=config.training_params.label_smoothing)
    criterion = WeightedCrossEntropyLoss(label_smoothing=config.training_params.label_smoothing)

    parameters = model.parameters()

    opt_name = config.training_params.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=config.training_params.lr,
            momentum=config.training_params.momentum,
            weight_decay=config.training_params.weight_decay,
            nesterov=config.training_params.nesterov,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=config.training_params.lr, momentum=config.training_params.momentum,
            weight_decay=config.training_params.weight_decay,
            eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=config.training_params.lr,
                                      weight_decay=config.training_params.weight_decay)
    else:
        raise RuntimeError(
            f"Invalid optimizer {config.opt}. Only SGD, RMSprop and AdamW are supported.")

    lr_scheduler = config.training_params.lr_scheduler.lower()

    if lr_scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.training_params.
                                                    schedule_kwargs.lr_step_size,
                                                    gamma=config.training_params.
                                                    schedule_kwargs.lr_gamma)
    elif lr_scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training_params.epochs - config.training_params.lr_warmup_epochs,
            eta_min=config.training_params.schedule_kwargs.lr_min
        )
    else:
        raise RuntimeError(
            f"lr_scheduler {config.lr_scheduler} not found.")

    if start_epoch > 0:  # restore the scheduler to that particular epoch state
        for _ in range(start_epoch):
            scheduler.step()

    train_len = len(train_dataset)
    real_train_target = train_dataset.targets

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.training_params.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True
                                               )

    # generate new images
    # if generator_param is not None:
    #     train_loader = update_train_loader(config, generator, generator_param, ratio, real_train_target,
    #                                        train_dataset, train_len, train_loader)

    #     # running through the discriminator to get alpha values

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.training_params.batch_size,
                                              shuffle=False,
                                              num_workers=config.workers,
                                              pin_memory=True
                                              )

    # Training
    LOGGER.info("Experiment config:")
    LOGGER.info(config)

    LOGGER.info(f"Start training from epoch {start_epoch} to {end_epoch}")
    start_time = time.time()
    best_acc = .0
    for epoch in range(start_epoch, end_epoch):
        epoch_start_time = time.time()

        if generator_param is not None:
            train_loader = update_train_loader(config, generator, generator_param, ratio,
                                               real_train_target,
                                               train_dataset, train_len, train_loader)

        train_one_epoch(model, criterion, optimizer, train_loader, epoch, D, D_augment_pipe, device)

        acc = evaluate(model, criterion, test_loader, epoch, device)

        if lr_scheduler == "steplr" or lr_scheduler == "cosineannealinglr":
            scheduler.step()

        epoch_end_time = time.time()
        epoch_time = time.strftime("%M:%S", time.gmtime(epoch_end_time - epoch_start_time))
        LOGGER.info(f"Epoch time: {epoch_time}")

    end_time = time.time()
    # Log the total time in terms of HH:MM:SS
    total_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    LOGGER.info(f"Total Training time: {total_time}")
    return model


def update_train_loader(config, generator, generator_param, ratio, real_train_target, train_dataset,
                        train_len, train_loader):
    # randomly select the training data index by ratio
    train_idx = torch.randperm(train_len)
    train_idx_selected = train_idx[:int(train_len * ratio)]
    train_idx_remains = train_idx[int(train_len * ratio):]
    # get labels for the selected index
    train_label_selected = [real_train_target[i] for i in train_idx_selected.tolist()]
    strat_generator_time = time.time()
    generated_dataset = generator_param['generator_fn'](generator,
                                                        train_label_selected,
                                                        **generator_param)
    generated_dataset.set_transform(train_dataset.transform)
    # concat the generated dataset with the remaining dataset
    train_subset = Subset(train_dataset, train_idx_remains)
    # train_subset = train_dataset
    new_train_dataset = ConcatIdentifiableDataset([train_subset, generated_dataset])
    generator.cpu()
    torch.cuda.empty_cache()
    LOGGER.info(f"Time to generate new images: {time.time() - strat_generator_time}")
    train_loader = torch.utils.data.DataLoader(new_train_dataset,
                                               batch_size=config.training_params.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True
                                               )
    return train_loader


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training",
                                     add_help=add_help)

    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--dist", type=bool, default=False,
                        help="Enable distributed training")
    parser.add_argument("--network_pkl", type=str, required=True,
                        help="Path to the network pickle file")
    parser.add_argument("--D_pkl", type=str, required=True,
                        help="Path to the discriminator pickle file")
    parser.add_argument("--D_config", type=str, default="configs/cifar10-D.yaml",
                        help="Path to the discriminator config file")
    parser.add_argument("--D_augment_config", type=str,
                        default="configs/cifar10-augment-D.yaml",
                        help="Path to the discriminator config file")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="Ratio of the training data to be used for generating new images")
    parser.add_argument("--generator_type", type=str, required=True,
                        help="Type of the generator to be used")
    parser.add_argument("--max_batch_size_for_gen", type=int, default=64,
                        help="Max batch size for generating new images")

    return parser


def main():
    dist.init()

    parser = get_args_parser()
    args = parser.parse_args()

    LOGGER = logging.getLogger(name='__main__')

    # name of config file
    args.start_time = str(datetime.now()).replace(" ", "-").replace(":", "-")
    name = os.path.basename(args.config).split('.')[0]
    name += f"_{args.generator_type}{args.ratio}"
    args.save_dir = os.path.join("results", name + "-" + args.start_time)
    os.makedirs(args.save_dir, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(args.save_dir, 'train.log'))
    file_handler.setLevel(logging.INFO)
    LOGGER.addHandler(file_handler)

    config = OmegaConf.load(args.config)

    generator_param = {}

    network_pkl = args.network_pkl
    if is_url(network_pkl):
        with load_pickle.open_url(network_pkl) as f:
            generator = pickle.load(f)['ema']
    else:
        with load_pickle.open_local_pkl(network_pkl) as f:
            generator = pickle.load(f)['ema']
    generator.eval()

    D_state = torch.load(args.D_pkl, map_location='cpu')['G_ema_state']
    D_config = OmegaConf.load(args.D_config)
    D = dnnlib.util.construct_class_by_name(**D_config)
    D.load_state_dict(D_state)
    D.eval()

    D_augment_pipe =dnnlib.util.construct_class_by_name(**OmegaConf.load(args.D_augment_config))

    generator.load_state_dict(torch.load(args.D_pkl)['G_ema_state'])

    if args.generator_type == "edm":
        generator_fn = generate_new_images_edm
    elif args.generator_type == "edm2":
        generator_fn = generate_new_images_edm2
    else:
        raise RuntimeError(f"Invalid generator type {args.generator_type}")

    generator_param['generator_fn'] = generator_fn
    generator_param['max_batch_size'] = args.max_batch_size_for_gen

    train_dataset, test_dataset = initialize_dataset(config)
    classifier = initialize_model(config)

    classify(classifier, generator, train_dataset, test_dataset, config,
             args.ratio, generator_param=generator_param, D=D, D_augment_pipe=D_augment_pipe)


if __name__ == "__main__":
    main()
