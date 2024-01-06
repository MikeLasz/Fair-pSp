from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import pandas as pd
from training.ethnicity_prediction import eth_to_code
from torchvision import transforms 
import torch 

class ImagesDataset(Dataset):

	def __init__(self, 
              source_root, 
              target_root, 
              opts, target_transform=None, 
              source_transform=None, 
              labels_path=None,
              col_name_label="race",
              col_name_file="file"):
		"""
		labels_path denotes the path to a csv wih columns col_name_label and
		col_name_file. 
		col_name_label denotes the column that stores the labels.
		col_name_file denotes the column that stores the file path. 
     	"""
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		self.labels = None
		if labels_path is not None:
			self.col_name_label = col_name_label
			self.col_name_file = col_name_file 
			self.labels = self.load_labels_csv(labels_path, col_name_label, col_name_file)
		if self.opts.use_augmentation:
			# Add RandomAffine (with translation, shearing, and scaling) and RandomHorizontalFLip to transform_source
			self.augmentation = transforms.Compose([
					transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10, scale=(0.9, 1.1)),
					transforms.RandomHorizontalFlip(p=0.5),
			])
	
	def load_labels_csv(self, path, col_name_label, col_name_file):
		"""Note: This only works for if the file-names are of the structure {base}/{index}.{ext}"""
		labels = pd.read_csv(path)
		labels["img_nr"] = labels[col_name_file].str.extract(r'(\d+)').astype(int)
		return labels[["img_nr", col_name_label]]

 
	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		to_path = self.target_paths[index]
		if self.labels is not None:
			img_nr = int(from_path.split("/")[-1].split(".")[0])
			label = self.labels[self.col_name_label][self.labels["img_nr"]==img_nr].item()
			label = eth_to_code(label)
		else:
			label = None
		to_im = Image.open(to_path).convert('RGB')

		
   
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im
		
		if self.opts.use_augmentation:
			from_im, to_im = self.augmentation(torch.stack([from_im, to_im]))
   
		return from_im, to_im, label

	def __getlabel__(self, index):
		from_path = self.source_paths[index]
  
		if self.labels is not None:
			img_nr = int(from_path.split("/")[-1].split(".")[0])
			label = self.labels[self.col_name_label][self.labels["img_nr"]==img_nr].item()
			label = eth_to_code(label)
		else:
			label = None

		return label
