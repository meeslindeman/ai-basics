def get_loaders(name: str, batch_size: int, num_workers: int = 2):
    name = name.lower()
    if name == "mnist":
        from .mnist import get_mnist_loaders
        return get_mnist_loaders(batch_size=batch_size, num_workers=num_workers)
    if name == "cifar10":
        from .cifar10 import get_cifar10_loaders
        return get_cifar10_loaders(batch_size=batch_size, num_workers=num_workers)
    raise ValueError(f"Unknown dataset: {name}")