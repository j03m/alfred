import torch


def set_device():
    if torch.cuda.is_available() and not is_debugger_active():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and not is_debugger_active():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def is_debugger_active():
    try:
        import pydevd  # Module used by PyCharm's debugger
        return True
    except ImportError:
        return False
