from torch import Tensor, nn
from typing import Tuple


class RepeatTarget(nn.Module):
    """Repeat the targets for ensemble training.

    Args:
        num_repeats: Number of times to repeat the targets.
    """

    def __init__(self, num_repeats: int) -> None:
        super().__init__()

        if not isinstance(num_repeats, int):
            raise ValueError(
                f"num_repeats must be an integer. Got {num_repeats}."
            )
        if num_repeats <= 0:
            raise ValueError(
                f"num_repeats must be greater than 0. Got {num_repeats}."
            )

        self.num_repeats = num_repeats

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        inputs, targets = batch
        return inputs, targets.repeat(self.num_repeats)
