import tonic.transforms
import torchvision


class DataTransform:
    def __init__(self, representation_fn, pre_augmentation_fn=None, post_augmentation_fn=None):
        self.representation_fn = representation_fn
        self.pre_augmentation_fn = pre_augmentation_fn
        self.post_augmentation_fn = post_augmentation_fn

        if self.pre_augmentation_fn is None:
            self.pre_augmentation_fn = tonic.transforms.Compose([])
        if self.post_augmentation_fn is None:
            self.post_augmentation_fn = torchvision.transforms.Compose([])

    def train_transform(self, events):
        x = self.pre_augmentation_fn(events)
        x = self.representation_fn(x)
        x = self.post_augmentation_fn(x)

        return x

    def val_transform(self, events):
        x = self.representation_fn(events)

        return x

    def pretrain_transform(self, events):
        x1 = events
        x2 = events

        x1 = self.pre_augmentation_fn(x1)
        x2 = self.pre_augmentation_fn(x2)

        x1 = self.representation_fn(x1)
        x2 = self.representation_fn(x2)

        x1 = self.post_augmentation_fn(x1)
        x2 = self.post_augmentation_fn(x2)

        return x1, x2
