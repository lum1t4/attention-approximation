from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import gc
import importlib
import logging
import math
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from typing_extensions import ParamSpec, TypeGuard, TypeVar

from attention_approximation.utils import PYTHON_VERSION, check_version, colorstr

torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic training
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ["KINETO_LOG_LEVEL"] = "5"  # suppress verbose PyTorch profiler output when computing FLOPs


LOGGER = logging.getLogger(__name__)

# Version checks (all default to version>=min_version)
TORCH_1_10 = check_version(torch.__version__, "1.10.0")
TORCH_1_9 = check_version(torch.__version__, "1.9.0")
TORCH_1_13 = check_version(torch.__version__, "1.13.0")
TORCH_2_0 = check_version(torch.__version__, "2.0.0")
TORCH_2_4 = check_version(torch.__version__, "2.4.0")

TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # torchvision version
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")


# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something."""
    initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank])
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[0])


def smart_inference_mode():
    """Applies torch.inference_mode() decorator if torch>=1.9.0 else torch.no_grad() decorator."""

    def decorate(fn):
        """Applies appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through
        else:
            return (torch.inference_mode if TORCH_1_9 else torch.no_grad)()(fn)

    return decorate


def convert_optimizer_state_dict_to_fp16(state_dict):
    """
    Converts the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

    This method aims to reduce storage size without altering 'param_groups' as they contain non-tensor data.
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict



def device_parse(
        device: Optional[str | torch.device] = None,
        batch: int = 0,
        newline: bool = False,
        verbose: bool = False
    ):
    """
    Selects the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object.
            Options are 'None', 'cpu', or 'cuda', or '0' or '0,1,2,3'. Defaults to an empty string, which auto-selects
            the first available GPU, or CPU if no GPU is available.
        batch (int, optional): Batch size being used in your model. Defaults to 0.
        newline (bool, optional): If True, adds a newline at the end of the log string. Defaults to False.
        verbose (bool, optional): If True, logs the device information. Defaults to True.

    Returns:
        (torch.device): Selected device.

    Raises:
        ValueError: If the specified device is not available or if the batch size is not a multiple of the number of
            devices when using multiple GPUs.

    Examples:
        >>> device_parse('cuda:0')
        device(type='cuda', index=0)

        >>> device_parse('cpu')
        device(type='cpu')

    Note:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """

    if isinstance(device, torch.device):
        return device
    
    if device is None:
        LOGGER.info("Auto-selecting device")
        device = "cuda" if torch.cuda.is_available() \
            else "mps" if torch.backends.mps.is_available() else "cpu"

    s = f"🚀 Python=({PYTHON_VERSION}) torch=({torch.__version__}) "
    device = str(device).lower()
    for remove in "cuda:", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == "cuda":
            device = "0"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1:  # multi-GPU
            if batch < 1:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    "please specify a valid batch size, i.e. batch=16."
                )
            if batch % n != 0:  # check batch_size is divisible by device_count
                raise ValueError(
                    f"'batch={batch}' must be a multiple of GPU count {n}. Try 'batch={batch // n * n}' or "
                    f"'batch={batch // n * n + n}', the nearest batch sizes evenly divisible by {n}."
                )
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # Prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)


def device_memory_used(device: torch.device) -> float:
    """Get accelerator memory utilization in GB."""
    if device.type == "mps":
        memory = torch.mps.driver_allocated_memory()
    elif device.type == "cuda":
        memory = torch.cuda.memory_reserved()
    else:
        return 0 # CPU memory is not tracked by PyTorch
    return memory / 1e9


def device_memory_clear(device: torch.device):
    """Clear accelerator memory on different platforms."""
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cpu":
        return
    else:
        torch.cuda.empty_cache()


def check_amp(device: torch.device) -> bool:
    import re
    prefix = colorstr("AMP: ")
    if device.type in {"cpu", "mps"}:
        return False  # AMP only used on CUDA devices
    else:
        # GPUs that have issues with AMP
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False
    return True



class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


def get_num_params(model: nn.Module):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model: nn.Module):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d() and BatchNorm2d() layers."""
    fuseddconv = (
        nn.ConvTranspose2d(
            deconv.in_channels,
            deconv.out_channels,
            kernel_size=deconv.kernel_size,
            stride=deconv.stride,
            padding=deconv.padding,
            output_padding=deconv.output_padding,
            dilation=deconv.dilation,
            groups=deconv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(deconv.weight.device)
    )

    # Prepare filters
    w_deconv = deconv.weight.clone().view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight.shape))

    # Prepare spatial bias
    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fuseddconv



def is_parallel(model):
    """Returns True if model is of type DP or DDP."""
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def de_parallel(model):
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {
        k: v
        for k, v in da.items()
        if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape
    }


def copy_attr(a, b, include=(), exclude=()):
    """Copies attributes from object 'b' to object 'a', with options to include/exclude certain attributes."""
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Create EMA."""
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1 - math.exp(-x / tau)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        """Update EMA parameters."""
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Updates attributes and saves stripped model with optimizer removed."""
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


class DistributedEvalSampler(Sampler):
    r"""
    source: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)  # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)  # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)



def init_seeds(seed: int = 0, deterministic: bool = True):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(
                True, warn_only=True
            )  # warn if deterministic is not possible
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            logging.warning(
                "WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training."
            )
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


T = TypeVar("T")
P = ParamSpec("P")

def rank_zero_only(fn: Callable[P, T], default: Optional[T] = None) -> Callable[P, Optional[T]]:
        @wraps(fn)
        def wrapped_fn(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
            if RANK is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            if RANK in {-1, 0}:
                return fn(*args, **kwargs)
            return default

        return wrapped_fn



def is_tensor(x: object, msg: Optional[str] = None, raises: bool = True) -> TypeGuard[Tensor]:
    """Check the input variable is a Tensor.

    Args:
        x: any input variable.
        msg: message to show in the exception.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        TypeException: if the input variable does not match with the expected and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 3)
        >>> is_tensor(x, "Invalid tensor")
        True

    """
    # TODO: Move to use typeguard here dropping support for JIT
    if not isinstance(x, Tensor):
        if raises:
            raise TypeError(f"Not a Tensor type. Got: {type(x)}.\n{msg}")
        return False
    return True


def check_shape(x: Tensor, shape: list[str], raises: bool = True) -> bool:
    """Check whether a tensor has a specified shape.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `Tensor`.

    Args:
        x: the tensor to evaluate.
        shape: a list with strings with the expected shape.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the input tensor is has not the expected shape and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> check_shape(x, ["B", "C", "H", "W"])  # implicit
        True

        >>> x = torch.rand(2, 3, 4, 4)
        >>> check_shape(x, ["2", "3", "H", "W"])  # explicit
        True

    """
    if "*" == shape[0]:
        shape_to_check = shape[1:]
        x_shape_to_check = x.shape[-len(shape) + 1 :]
    elif "*" == shape[-1]:
        shape_to_check = shape[:-1]
        x_shape_to_check = x.shape[: len(shape) - 1]
    else:
        shape_to_check = shape
        x_shape_to_check = x.shape

    if len(x_shape_to_check) != len(shape_to_check):
        if raises:
            raise TypeError(f"{x} shape must be [{shape}]. Got {x.shape}")
        else:
            return False

    for i in range(len(x_shape_to_check)):
        # The voodoo below is because torchscript does not like
        # that dim can be both int and str
        dim_: str = shape_to_check[i]
        if not dim_.isnumeric():
            continue
        dim = int(dim_)
        if x_shape_to_check[i] != dim:
            if raises:
                raise TypeError(f"{x} shape must be [{shape}]. Got {x.shape}")
            else:
                return False
    return True
