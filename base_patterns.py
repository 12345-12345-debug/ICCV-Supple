from PIL import Image
from augly.image.transforms import *
import torchvision.transforms as transforms
import random
import pandas as pd

class RandomRotate:
    def __init__(self, degrees = [0,360], name = 'RandomRotate'):
        self.degrees = degrees
        self.name = name

    def __call__(self, x):
        degree = random.uniform(self.degrees[0], self.degrees[1])
        x = Rotate(degrees = degree)(x)
        return x

class HoriFlip:
    def __init__(self, name = 'HoriFlip'):
        self.name = name

    def __call__(self, x):
        return HFlip()(x)

class RandomBright:
    def __init__(self, factors = [0.2, 2], name = 'RandomBright'):
        self.factors = factors
        self.name = name

    def __call__(self, x):
        factor = random.uniform(self.factors[0], self.factors[1])
        x = Brightness(factor = factor)(x)
        return x

class RandomContrast:
    def __init__(self, factors = [0.2, 4], name = 'RandomContrast'):
        self.factors = factors
        self.name = name

    def __call__(self, x):
        factor = random.uniform(self.factors[0], self.factors[1])
        x = Contrast(factor = factor)(x)
        return x

class RandomOpacity:
    def __init__(self, levels = [0.6, 1], name = 'RandomOpacity'):
        self.levels = levels
        self.name = name

    def __call__(self, x):
        level = random.uniform(self.levels[0], self.levels[1])
        x = Opacity(level = level)(x)
        return x

class RandomOverlayEmoji:
    def __init__(self, path = '/raid/DGICD/data/emoji/', opacity=[0.2, 1], emoji_size=[0.2, 1], x_pos=[0, 0.5], y_pos=[0, 0.5], name = 'RandomOverlayEmoji'):
        self.path = path
        self.opacity = opacity
        self.emoji_size = emoji_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.name = name

    def __call__(self, x):
        emoji_path = self.path + random.choice(os.listdir(self.path))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        emoji_size = random.uniform(self.emoji_size[0], self.emoji_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        x = OverlayEmoji(emoji_path = emoji_path,
                         opacity = opacity,
                         emoji_size = emoji_size,
                         x_pos = x_pos,
                         y_pos = y_pos)(x)
        return x

class RandomOverlayImage:
    def __init__(self, path = '/raid/DGICD/data/train_0/', opacity=[0.6, 1], overlay_size=[0.5, 1], x_pos=[0, 0.5], y_pos=[0, 0.5], name = 'RandomOverlayImage'):
        self.path = path
        self.opacity = opacity
        self.overlay_size = overlay_size
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.name = name

    def __call__(self, x):
        bg = Image.open(self.path + random.choice(os.listdir(self.path)))
        opacity = random.uniform(self.opacity[0], self.opacity[1])
        overlay_size = random.uniform(self.overlay_size[0], self.overlay_size[1])
        x_pos = random.uniform(self.x_pos[0], self.x_pos[1])
        y_pos = random.uniform(self.y_pos[0], self.y_pos[1])
        bg = OverlayImage(overlay = x,
                         opacity = opacity,
                         overlay_size = overlay_size,
                         x_pos = x_pos,
                         y_pos = y_pos)(bg)
        return bg


class RandomPad:
    def __init__(self, w_factors = [0, 0.5], h_factors = [0, 0.5], color_1s = [0,255], color_2s = [0,255], color_3s = [0,255], name = 'RandomPad'):
        self.w_factors = w_factors
        self.h_factors = h_factors
        self.color_1s = color_1s
        self.color_2s = color_2s
        self.color_3s = color_3s
        self.name = name

    def __call__(self, x):
        w_factor = random.uniform(self.w_factors[0], self.w_factors[1])
        h_factor = random.uniform(self.h_factors[0], self.h_factors[1])
        color_1 = random.randint(self.color_1s[0], self.color_1s[1])
        color_2 = random.randint(self.color_2s[0], self.color_2s[1])
        color_3 = random.randint(self.color_3s[0], self.color_3s[1])
        x = Pad(w_factor = w_factor, h_factor = h_factor, color = (color_1, color_2, color_3))(x)
        return x

class RandomPerspectiveTransform:
    def __init__(self, sigmas = [10, 50], name = 'RandomPerspectiveTransform'):
        self.sigmas = sigmas
        self.name = name

    def __call__(self, x):
        sigma = random.uniform(self.sigmas[0], self.sigmas[1])
        x = PerspectiveTransform(sigma=sigma)(x)
        return x

class RandomPixelization:
    def __init__(self, ratios = [0.1, 1], name = 'RandomPixelization'):
        self.ratios = ratios
        self.name = name

    def __call__(self, x):
        ratio = random.uniform(self.ratios[0], self.ratios[1])
        x = Pixelization(ratio = ratio)(x)
        return x

class RandomShufflePixels:
    def __init__(self, factors = [0.1, 0.5], name = 'RandomShufflePixels'):
        self.factors = factors
        self.name = name

    def __call__(self, x):
        factor = random.uniform(self.factors[0], self.factors[1])
        x = ShufflePixels(factor = factor)(x)
        return x
