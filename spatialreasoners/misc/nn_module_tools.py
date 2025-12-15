from torch import nn


def convert_to_buffer(
    module: nn.Module, 
    persistent: bool = True
) -> None:
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def zero_init(
    module: nn.Module
) -> None:
    for p in module.parameters():
        nn.init.zeros_(p)


def constant_init(module: nn.Module, val: float | int, bias: float | int = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def requires_grad(model: nn.Module, flag: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def freeze(m: nn.Module | nn.Parameter, eval: bool = True) -> None:
    if isinstance(m, nn.Parameter):
        m.requires_grad = False
    else:
        requires_grad(m, False)
        if eval:
            m.eval()


def unfreeze(m: nn.Module | nn.Parameter, train: bool = True) -> None:
    if isinstance(m, nn.Parameter):
        m.requires_grad = True
    else:
        requires_grad(m, True)
        if train:
            m.train()
