import torch
from torch import nn


def fakescore_discriminator_share_encoder_loss(self, fake_score: torch.nn.Module,
                                               images,  # generated images
                                               labels=None,
                                               augment_pipe: callable = None):
    rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
    sigma = (rnd_normal * self.P_std + self.P_mean).exp()

    if augment_pipe is not None:
        y_augmented, augment_labels = augment_pipe(images)
    else:
        y_augmented = images
        augment_labels = torch.zeros(images.shape[0], 9).to(images.device)

    n = torch.randn_like(y_augmented) * sigma  # noise [fake_batch]
    y_fake, logit_fake = fake_score(y_augmented + n, sigma, labels,
                                    augment_labels=augment_labels,
                                    return_flag='encoder_decoder')

    with torch.no_grad():
        weight_factor = abs(y_augmented - y_fake).to(torch.float32).mean(dim=[1, 2, 3],
                                                                         keepdim=True).clip(
            min=0.00001)


    nan_mask = (torch.isnan(y_augmented).flatten(start_dim=1).any(dim=1) |
                torch.isnan(y_fake).flatten(start_dim=1).any(dim=1))

    if nan_mask.any():
        # Invert the nan_mask to get a mask of samples without NaNs
        non_nan_mask = ~nan_mask
        # Filter out samples with NaNs from y_real and y_fake
        logit_fake = logit_fake[non_nan_mask]

        weight = weight[non_nan_mask]

    fake_labels = torch.zeros_like(logit_fake)
    bce_loss = nn.BCEWithLogitsLoss()
    loss_fake = bce_loss(logit_fake, fake_labels)
    loss_D = weight * (loss_fake)

    return loss_D