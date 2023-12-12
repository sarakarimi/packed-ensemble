from pathlib import Path

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import DeepEnsembles
from torch_uncertainty.datamodules import CIFAR10DataModule

# This code is take from the main repository of
# torch-uncertainty package: https://github.com/ENSTA-U2IS/torch-uncertainty


args = init_args(DeepEnsembles, CIFAR10DataModule)
if args.root == "./data/":
    root = Path(__file__).parent.absolute().parents[2]
else:
    root = Path(args.root)

# args.arch = 50
# args.max_epochs = 200
args.num_estimators = 4
args.accelerator = "gpu"
args.device = 1

net_name = f"de-{args.backbone}-cifar10"

# datamodule
args.root = str(root / "data")
dm = CIFAR10DataModule(**vars(args))

# model
args.task = "classification"
model = DeepEnsembles(
    **vars(args),
    num_classes=dm.num_classes,
    in_channels=dm.num_channels,
)

args.test = -1

if args.test is not None:
    for seed in [1000, 2000, 3000, 4000, 5000]:
        args.seed = seed
        cli_main(model, dm, root, net_name, args)
else:
    cli_main(model, dm, root, net_name, args)