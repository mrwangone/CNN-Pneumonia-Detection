import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm


def compute_normalization_values(data_dir, image_size=224, batch_size=32):
    """
    Compute mean and std of your dataset for normalization.

    Args:
        data_dir: Path to training data directory
        image_size: Size to resize images to
        batch_size: Batch size for processing

    Returns:
        means: Channel-wise mean
        stds: Channel-wise standard deviation
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # Scales data into [0,1]
    ])

    # Load training data only
    train_dataset = datasets.ImageFolder(
        f"{data_dir}/train",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize variables
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    # Computing mean
    print("Computing mean...")
    for images, _ in tqdm(train_loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_images += batch_samples

    mean = mean / total_images

    # Computing std
    print("Computing std...")
    for images, _ in tqdm(train_loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.view(3, 1)) ** 2).mean(2).sum(0)

    std = torch.sqrt(std / total_images)

    return mean.tolist(), std.tolist()


def main():
    data_dir = "./dataset"
    mean, std = compute_normalization_values(data_dir)

    print("\nDataset normalization values:")
    print(f"Mean: {[round(m, 4) for m in mean]}")
    print(f"Std: {[round(s, 4) for s in std]}")

    print("\nUse these values in your transforms like this:")
    print("transforms.Normalize("
          f"mean={[round(m, 4) for m in mean]}, "
          f"std={[round(s, 4) for s in std]})")


if __name__ == "__main__":
    main()
