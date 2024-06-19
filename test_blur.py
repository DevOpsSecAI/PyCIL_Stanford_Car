from torchvision import transforms
from PIL import Image
import argparse

def main():
    args = setup_parser().parse_args()
    path = args.path
    img = Image.open(path)
    trf =  transforms.Compose([
                transforms.RandomAffine(25, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness = 0.3, saturation = 0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=1),  # Apply Gaussian blur with random probability
            ])
    img = trf(img)
    img.save("blur.jpg")

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--path', type=str,
                        help='Image file.')

    return parser


    
if __name__ == '__main__':
    main()
    