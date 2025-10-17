from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


def get_cosine_scheduler_with_warmup(optimizer, num_steps, warmup_steps, min_lr):
    # Warmup Scheduler: Linearly increase LR from 0 to initial_lr
    warmup_scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: step / warmup_steps if step < warmup_steps else 1.0,
    )

    # Cosine Annealing Scheduler: Decrease LR from initial_lr to min_lr
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_steps - warmup_steps, eta_min=min_lr
    )

    # Combine Schedulers
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler
