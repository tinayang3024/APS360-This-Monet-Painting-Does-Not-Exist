import glob
import random
import torch
import torchvision
from PIL import Image

class customDataset(torch.utils.data.Dataset):
  """ Create custom dataset that return pair of A and B data
      Example:  
        dataset = customDataset(root=root+'impressionism/', mode='train', dataset_size=100) 
  """
  def __init__(self, root, mode, dataset_size):
    self.trans = torchvision.transforms.Compose([
      torchvision.transforms.Resize(256),
      torchvision.transforms.CenterCrop(256),
      torchvision.transforms.ToTensor(),
    ])
    self.A_path = root + mode + 'A' 
    self.B_path = root + mode + 'B'
    self.A_data = []
    self.B_data = []
    for i, filename in enumerate(glob.glob(self.A_path+'/*.jpg')):
      if (i > dataset_size):
        break
      self.A_data.append(filename)
    for i, filename in enumerate(glob.glob(self.B_path+'/*.jpg')):
      if (i > dataset_size):
        break
      self.B_data.append(filename)
    self.size = dataset_size

  def __getitem__(self, idx):
    """ return pair of A and B data in the format:
        {'A': {'data': imgA, 'path': pathA}, 'B': {'data': imgB, 'path': pathB}}
    """
    if idx >= len(self.A_data):
      idx = idx % len(self.A_data)
    randidx = random.randint(0, len(self.B_data) - 1)
    return {'A': {'data': self.trans(Image.open(self.A_data[idx])), 'path': self.A_data[idx]} \
            , 'B': {'data': self.trans(Image.open(self.B_data[randidx])), 'path': self.B_data[randidx]}}

  def __len__(self):
    return self.size
