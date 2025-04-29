import os
import torch
import numpy as np
from PIL import Image

def tensor_to_image(
    tensor: torch.Tensor,
    input_range: str = "auto",
    format: str = "RGB"
) -> Image.Image:
    """
    Convert a PyTorch tensor to PIL Image, supporting both [0,1] and [-1,1] ranges.

    Args:
        tensor (torch.Tensor): Input tensor in one of these shapes:
            - (C, H, W)          # Single image
            - (B, C, H, W)       # Batch of images (returns first image)
            - (H, W)             # Grayscale
        input_range (str): Value range detection mode:
            - 'auto': Automatically detect range (default)
            - '01': Force [0,1] range
            - '-11': Force [-1,1] range
        format (str): Output color format - 'RGB' or 'L' (grayscale)

    Returns:
        PIL.Image.Image: Resulting image in specified format

    Raises:
        TypeError: If input is not a torch.Tensor
        ValueError: For invalid input ranges or tensor shapes

    Example:
        >>> # GAN-generated example
        >>> gan_tensor = torch.randn(3, 256, 256).clamp(-1, 1)
        >>> img = tensor_to_image(gan_tensor, input_range='-11')
        >>> img.save("gan_output.jpg")
    """
    # Type checking
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    # Dimension handling
    if tensor.dim() == 4:
        tensor = tensor[0]  # Get first image from batch
    elif tensor.dim() not in [2, 3]:
        raise ValueError(f"Invalid tensor shape: {tensor.shape}")

    # Range detection logic
    if input_range == "auto":
        t_min, t_max = tensor.min().item(), tensor.max().item()
        if t_min >= -1.0 and t_max <= 1.0:
            input_range = '-11' if (t_min < -0.5 or t_max > 0.5) else '01'
        else:
            raise ValueError(f"Auto-detection failed for tensor range [{t_min:.2f}, {t_max:.2f}]")

    # Normalization
    tensor = tensor.detach().cpu()
    if input_range == '-11':
        tensor = (tensor + 1.0) / 2.0  # [-1,1] -> [0,1]
    elif input_range != '01':
        raise ValueError(f"Invalid input_range: {input_range}")
    
    # Value clamping
    tensor = torch.clamp(tensor, 0.0, 1.0)

    # Channel handling
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # Add channel dim

    # Color format conversion
    if format == "L" and tensor.size(0) == 3:
        tensor = tensor.mean(dim=0, keepdim=True)  # RGB to grayscale
    elif format == "RGB" and tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)  # Grayscale to RGB

    # Array conversion
    np_array = tensor.numpy()
    np_array = np.transpose(np_array, (1, 2, 0))  # CHW -> HWC
    np_array = (np_array * 255).astype(np.uint8)

    # PIL image creation
    if format == "L":
        return Image.fromarray(np_array.squeeze(), mode="L")
    return Image.fromarray(np_array, mode="RGB")


def visualize_A2B(tensorA,tensorB,fake_tensorB,filename,log_path):
    save_path = os.path.join(log_path,filename)
    imgA = tensor_to_image(tensorA)
    imgB = tensor_to_image(tensorB)
    fake_imgB = tensor_to_image(fake_tensorB)
    images = [imgA,imgB,fake_imgB]
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_img = Image.new(images[0].mode, (total_width, max_height))
    
    # 依次粘贴图像
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    new_img.save(save_path)


@torch.no_grad()
def save_single_image(image, save_path, file_name, to_normal=True):
    image = image.detach().clone()
    if to_normal:
        image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(image)
    im.save(os.path.join(save_path, file_name))


@torch.no_grad()
def get_image_grid(batch, grid_size=4, to_normal=True):
    batch = batch.detach().clone()
    image_grid = make_grid(batch, nrow=grid_size)
    if to_normal:
        image_grid = image_grid.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image_grid = image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return image_grid