from pathlib import Path

from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.optimization_procedures import get_procedure

# This code is take from the main repository of
# torch-uncertainty package: https://github.com/ENSTA-U2IS/torch-uncertainty

root = Path(__file__).parent.absolute().parents[1]

args = init_args(ResNet, CIFAR100DataModule)

args.version = "batched"
# args.arch = 50
# args.max_epochs = 200
args.num_estimators = 4
args.accelerator = "gpu"
args.device = 1


net_name = f"{args.version}-resnet{args.arch}-cifar100"

# datamodule
args.root = str(root / "data")
dm = CIFAR100DataModule(**vars(args))

# model
model = ResNet(
    num_classes=dm.num_classes,
    in_channels=dm.num_channels,
    loss=nn.CrossEntropyLoss,
    optimization_procedure=get_procedure(
        f"resnet{args.arch}", "cifar100", args.version
    ),
    style="cifar",
    **vars(args),
)

if args.test is not None:
    for seed in [1000, 2000, 3000, 4000, 5000]:
        args.seed = seed
        cli_main(model, dm, root, net_name, args)
else:
    cli_main(model, dm, root, net_name, args)
