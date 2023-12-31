from pathlib import Path

from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import get_procedure

# This code is take from the main repository of
# torch-uncertainty package: https://github.com/ENSTA-U2IS/torch-uncertainty

root = Path(__file__).parent.absolute().parents[1]

args = init_args(ResNet, CIFAR10DataModule)

args.version = "batched"
args.arch = 50
# args.max_epochs = 200
args.num_estimators = 4
args.accelerator = "gpu"
args.device = 1

net_name = f"{args.version}-resnet{args.arch}-cifar10"

# datamodule
args.root = str(root / "data")
dm = CIFAR10DataModule(**vars(args))

# model
model = ResNet(
    num_classes=dm.num_classes,
    in_channels=dm.num_channels,
    loss=nn.CrossEntropyLoss,
    optimization_procedure=get_procedure(
        f"resnet{args.arch}", "cifar10", args.version
    ),
    style="cifar",
    **vars(args),
)

cli_main(model, dm, root, net_name, args)
