import math


def linear_decay(initial_lr, step, total_steps):
    if step >= total_steps:
        return 0.0
    return initial_lr * (1 - step / total_steps)


def cosine_annealing(initial_lr, step, total_steps, min_lr=0.0):
    if step >= total_steps:
        return min_lr
    return min_lr + (initial_lr - min_lr) * \
           (1 + math.cos(math.pi * step / total_steps)) / 2


def constant_lr(initial_lr, step, total_steps):
    return initial_lr


def step_decay(initial_lr, step, total_steps, step_size=30, gamma=0.1):
    return initial_lr * (gamma ** (step // step_size))


def get_lr_scheduler(config, total_steps):
    scheduler_type = config.get('type', 'linear_decay')
    
    steps = config.get('total_steps', total_steps)
    if isinstance(steps, str):
        steps = total_steps
    
    if scheduler_type == 'linear_decay':
        return lambda initial_lr, step: linear_decay(initial_lr, step, steps)
    
    elif scheduler_type == 'cosine':
        min_lr = config.get('min_lr', 0.0)
        return lambda initial_lr, step: cosine_annealing(initial_lr, step, steps, min_lr)
    
    elif scheduler_type == 'constant':
        return lambda initial_lr, step: constant_lr(initial_lr, step, steps)
    
    elif scheduler_type == 'step':
        step_size = config.get('step_size', 30)
        gamma = config.get('gamma', 0.1)
        return lambda initial_lr, step: step_decay(initial_lr, step, steps, step_size, gamma)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

