"""
Module responsible for data augmentation constants and configuration.  
"""

import torch as ch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as TVF


class MakeCircular(ch.nn.Module):
    """ Masks (sets to 0) all pixels outside of the circle/ellipse
    that is inscribed in the image. Intended to be used for
    rotationally-augmented datasets.

    Args:
        tensor: [torch.Tensor] or iterable thereof
    Returns:
        masked tensor(s)
    """
    def __init__(self):
        super().__init__()
        self.mask = None

    def create_mask(self, h, w):
        # only gets executed once, no need to be efficient
        mask = np.ones([h, w])
        r_h, r_w = h // 2, w // 2
        for i in range(h):
            for j in range(w):
                d_i = i - r_h
                d_j = j - r_w
                # check if outside of circle/ellipse
                if d_i ** 2 / r_h ** 2 + d_j ** 2 / r_w ** 2 > 1:
                    mask[i][j] = 0.
        # add empty dims for channels
        return ch.from_numpy(mask).float().unsqueeze(0)

    def make_circular(self, tensor):
        if self.mask is None:
            # assert len(img.shape) == 3, "data must be in c, h, w format"
            self.mask = self.create_mask(tensor.size(-2), tensor.size(-1))
        return (self.mask * tensor)  # batch and channel dims are broadcasted

    def forward(self, tensor):
        if isinstance(tensor, ch.Tensor):
            return self.make_circular(tensor)
        else:
            # assert isinstance(tensor, Iterable)
            return ch.stack([self.make_circular(ts) for ts in tensor])

    def __repr__(self):
        return "mask (set to 0) pixels outside of the inscribed circle/ellipse"


class MultipleRandomRotations(ch.nn.Module):
    """ Similar to transforms.RandomRotation, but returns
    multiple rotations instead.

    Args:
        img: [PIL Image]
        num_rots: [int]
        degree: [float] rotations sampled in the range (-degree, degree)
    Returns:
        tuple containing the original image, together with num_rots
        rotated images (note that the tuple is of size num_rots + 1)
    """
    def __init__(self, num_rots, degree, adv_class=-1,
                 resample=Image.BICUBIC, expand=False,
                 center=None, fill=None):
        super().__init__()
        self.degree = degree
        self.center = center
        self.resample = resample
        self.expand = expand
        self.fill = fill
        self.num_rots = num_rots

    @staticmethod
    def get_params(degree):
        angle = float(ch.empty(1).uniform_(float(-degree),
                                           float(degree)).item())
        return angle

    def forward(self, img):
        rot_imgs = []
        for i in range(self.num_rots + 1):
            angle = self.get_params(self.degree) if i > 0 else 0.
            rot_img = TVF.rotate(img, angle, self.resample, self.expand,
                                 self.center, self.fill)
            rot_imgs.append(rot_img)

        return tuple(rot_imgs)


class NoneTransform(object):
    """ Convenience class, does nothing to the image

    Args:
        image in, image out, nobody gets hurt
    """
    def __call__(self, image):
        return image


# lighting transform
# https://git.io/fhBOc

IMAGENET_PCA = {
    'eigval': ch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': ch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


# Special transforms for ImageNet(s)
TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'],
                 IMAGENET_PCA['eigvec'])
    ])
"""
Standard training data augmentation for ImageNet-scale datasets: Random crop,
Random flip, Color Jitter, and Lighting Transform (see https://git.io/fhBOc)
"""

TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
"""
Standard test data processing (no augmentation) for ImageNet-scale datasets,
Resized to 256x256 then center cropped to 224x224.
"""

# Data Augmentation defaults
TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25, .25, .25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
"""
Generic training data transform, given image side length does random cropping,
flipping, color jitter, and rotation. Called as, for example,
:meth:`robustness.data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32)` for CIFAR-10.
"""

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
"""
Generic test data transform (no augmentation) to complement
:meth:`robustness.data_augmentation.TEST_TRANSFORMS_DEFAULT`, takes in an image
side length.
"""


# Special rotationally-augmented transforms for ImageNet(s)
def get_rot_transforms(num_rots, resampling=Image.BICUBIC, make_circ=True):
    jitter = transforms.ColorJitter(brightness=0.1,
                                    contrast=0.1,
                                    saturation=0.1)
    lighting = Lighting(0.05, IMAGENET_PCA['eigval'],
                        IMAGENET_PCA['eigvec'])

    TRAIN_ROT_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        MultipleRandomRotations(num_rots, degree=180., resample=resampling),
        transforms.Lambda(lambda imgs: tuple([jitter(i) for i in imgs])),
        transforms.Lambda(lambda imgs:
                          tuple([transforms.ToTensor()(i) for i in imgs])),
        transforms.Lambda(lambda tensors:
                          ch.stack([lighting(t) for t in tensors])),
        MakeCircular() if make_circ else NoneTransform()
        ])

    TEST_ROT_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        MultipleRandomRotations(num_rots, degree=180., resample=resampling),
        transforms.Lambda(lambda imgs:
                          ch.stack([transforms.ToTensor()(i) for i in imgs])),
        MakeCircular() if make_circ else NoneTransform()
        ])
    return TRAIN_ROT_TRANSFORMS_IMAGENET, TEST_ROT_TRANSFORMS_IMAGENET
