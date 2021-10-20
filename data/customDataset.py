import glob
import random
import torch
import torchvision
from PIL import Image

class customDataset(torch.utils.data.Dataset):
  """ Create custom dataset that return pair of A and B data
      Example:  
        dataset = customDataset(root=root+'impressionism/', mode='train') 
  """
  def __init__(self, root, mode):
    trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor(),
    ])
    self.A_path = root + mode + 'A' 
    self.B_path = root + mode + 'B'
    self.A_data = []
    self.B_data = []
    for filename in glob.glob(self.A_path+'/*.jpg'):
      self.A_data.append(filename)
    for filename in glob.glob(self.B_path+'/*.jpg'):
      self.B_data.append(filename)
    self.size = max(len(self.A_data), len(self.B_data))

  def __getitem__(self, idx):
    """ return pair of A and B data in the format:
        {'A': {'data': imgA, 'path': pathA}, 'B': {'data': imgB, 'path': pathB}}
    """
    if idx >= len(self.A_data):
      idx = idx % len(self.A_data)
    randidx = random.randint(0, len(self.B_data) - 1)
    return {'A': {'data': Image.open(self.A_data[idx]), 'path': self.A_data[idx]} \
            , 'B': {'data': Image.open(self.B_data[randidx]), 'path': self.B_data[randidx]}}

  def __len__(self):
    return self.size
