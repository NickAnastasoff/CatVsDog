from torchvision import transforms, datasets, models
import torchvision
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def __main__():
  # Data augmentation and normalization for training
  # Just normalization for validation
  data_transforms = {
      'train': transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  # We want the trains that have a  lot of deviation
  # but we also want the test set to be as easy to predict as possible

  data_dir = 'data'

  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                                            
                    for x in ['train', 'test']}
  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
                for x in ['train', 'test']}
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
  class_names = image_datasets['train'].classes

  def imshow(inp, title=None):
      """Display image for Tensor."""
      inp = inp.numpy().transpose((1, 2, 0))
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      inp = std * inp + mean
      inp = np.clip(inp, 0, 1)
      plt.imshow(inp)
      if title is not None:
          plt.title(title)
      plt.savefig("output.png")  # This saves the image to a file
      plt.show()
      print(title)



  # Get a batch of training data
  inputs, classes = next(iter(dataloaders['train']))
  for input in inputs:
      print(f"Image size: {input.shape}")

  # Make a grid from batch
  out = torchvision.utils.make_grid(inputs)
  imshow(out, title=[class_names[x] for x in classes])

if __name__ == "__main__":
  __main__()