import numpy as np
import torch


class _CVTransformBase(object):
    def __init__(self, name):
        self.name = name


class Identity(_CVTransformBase):
    def __init__(self):
        super(self.__class__, self).__init__("identity")

    def __call__(self, image):
        return image


class ReverseColorChannels(_CVTransformBase):
    def __init__(self, format='HWC'):
        super(self.__class__, self).__init__("reverse color channels")
        if format == 'HWC':
            self.dim = 2
        elif format == 'CHW':
            self.dim = 0
        else:
            raise ValueError("Invalid format '{}' provided".format(format))

    def __call__(self, image):
        return np.flip(image, self.dim)


class Normalize(_CVTransformBase):
    def __init__(self, norm_factor, mean, std):
        super(self.__class__, self).__init__("normalize")
        self.__norm_factor = np.float32(norm_factor)
        self.__mean = np.array([[mean]], np.float32)
        self.__std = np.array([[std]], np.float32)

    def __call__(self, image):
        normed_image = image.astype(np.float32) / self.__norm_factor
        normed_image = (normed_image - self.__mean) / self.__std
        return normed_image


class ReverseNormalize(_CVTransformBase):
    def __init__(self, norm_factor, mean, std, dtype=np.uint8):
        super(self.__class__, self).__init__("reverse normalize")
        self.__norm_factor = np.float32(norm_factor)
        self.__mean = np.array([[mean]], np.float32)
        self.__std = np.array([[std]], np.float32)
        self.__dtype = dtype

    def __call__(self, image):
        rnormed_image = (image * self.__std) + self.__mean
        rnormed_image = rnormed_image * self.__norm_factor
        return rnormed_image.astype(self.__dtype)


class ToTorchTensor(_CVTransformBase):
    def __init__(self, format='HWC', dtype=None):
        super(self.__class__, self).__init__("to torch tensor")
        assert format in ['HWC', 'CHW']
        self.format = format
        self.dtype = dtype

    def __call__(self, image):
        tensor = torch.from_numpy(np.ascontiguousarray(image))

        if tensor.ndimension() == 3 and self.format == 'CHW':
            tensor = tensor.permute(2, 0, 1)

        if self.dtype is not None:
            return tensor.to(dtype=self.dtype)
        else:
            return tensor


class Compose(_CVTransformBase):
    def __init__(self, transforms):
        super(self.__class__, self).__init__("composition")
        self.__transforms = transforms

    def __call__(self, image):
        for transform in self.__transforms:
            image = transform(image)
        return image


class BatchImageTransform(_CVTransformBase):
    def __init__(self, transform):
        super(self.__class__, self).__init__("batch image transform")
        self.__transform = transform

    def __call__(self, *images):
        return [self.__transform(image) for image in images]
