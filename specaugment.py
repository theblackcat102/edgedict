import torch
import random


class FrequencyMask(torch.nn.Module):
    """
    Implements frequency masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)

      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     FrequencyMask(max_width=10, use_mean=False),
        >>> ])

    """

    def __init__(self, max_width, num_masks, use_mean=True):
        super().__init__()
        self.max_width = max_width
        self.num_masks = num_masks
        self.use_mean = use_mean

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (T, H) where the frequency mask is to be applied.

        Returns:
            Tensor: Transformed image with Frequency Mask.
        """
        if self.use_mean:
            fill_value = tensor.mean()
        else:
            fill_value = 0
        for _ in range(self.num_masks):
            start = random.randrange(0, tensor.shape[1])
            end = start + random.randrange(0, self.max_width)
            tensor[:, start:end] = fill_value
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class TimeMask(torch.nn.Module):
    """
    Implements time masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)

      Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     TimeMask(max_width=10, use_mean=False),
        >>> ])

    """

    def __init__(self, max_width, num_masks, use_mean=True):
        super().__init__()
        self.max_width = max_width
        self.num_masks = num_masks
        self.use_mean = use_mean

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (T, H) where the time mask is to be applied.

        Returns:
            Tensor: Transformed image with Time Mask.
        """
        if self.use_mean:
            fill_value = tensor.mean()
        else:
            fill_value = 0
        for _ in range(self.num_masks):
            start = random.randrange(0, tensor.shape[0])
            end = start + random.randrange(0, self.max_width)
            tensor[start:end, :] = fill_value
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string
