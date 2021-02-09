import os
import torch.utils.data as data
from PIL import Image
import random
import numpy as np
import json
import sys
from collections import OrderedDict

class shapenet(data.Dataset):
	def __init__(self, hyperparameters, transform=None):
		self.root_dir = hyperparameters["dataset_root"]
		self.classes = hyperparameters["classes"].split(",")
		self.transform = transform
		self.input_dim = hyperparameters["input_dim"]
		#load cat info
		cats = json.load(open(hyperparameters["cat_json"]))
		cats = OrderedDict(sorted(cats.items(), key=lambda x: x[0]))
		imgs = []
		labels = []
		_ind = 0
		n_class = 0
		n_model = 0
		n_views = hyperparameters["n_views"]
		for _dir in os.listdir(self.root_dir):
			if _dir not in cats:
				continue
			if self.classes and cats[_dir]['cat'] not in self.classes:
				continue
			_dir_path = os.path.join(self.root_dir, _dir)
			if not os.path.isdir(_dir_path):
				continue
			n_class += 1
			class_start_ind = _ind
			models = self.get_model_names(_dir_path)
			for model in models:
				ok, img_names = self.get_img_names(_dir, model)
				if not ok:
					continue
				n_model += 1
				if hyperparameters["randome_num_views"]:
					curr_n_views = np.random.randint(min(n_views, len(img_names))) + 1
				else:
					curr_n_views = min(n_views, len(img_names))
				image_inds = np.random.choice(len(img_names), curr_n_views)
				for ind in image_inds:
					img_path = self.get_img_path(_dir, model, img_names[ind])
					if os.path.isfile(img_path):
						imgs.append(img_path)
						labels.append(cats[_dir]['cat'])
						_ind += 1
				class_end_ind = _ind - 1
			print('load %d images from %d models from class=%s, id=%s' %(
				class_end_ind - class_start_ind + 1, len(models), cats[_dir]['cat'], _dir))
		self.imgs = imgs
		self.labels = labels
		self.dataset_size = len(imgs)
		print('totally load %d images from %d models from %d classes' %(len(self.imgs), n_model, n_class))

	def __getitem__(self, index):
		data = self.load_img(self.imgs[index], self.input_dim)
		label = self.labels[index]
		return {"data": data, "label": label}

	def load_img(self, img_name, input_dim):
		# discard alpha channel
		img = Image.open(img_name).convert('RGB')
		img = self.transform(img)
		if input_dim == 1:
			img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
			img = img.unsqueeze(0)
		return img

	def __len__(self):
		return self.dataset_size

	def get_img_path(self, cat_id, model_id, img_name):
		return os.path.join(self.root_dir, cat_id, model_id, 'rendering', img_name)

	def get_img_names(self, cat_id, model_id):
		result = []
		meta_path = os.path.join(self.root_dir, cat_id, model_id, 'rendering/rendering_metadata.txt')
		img_names = os.path.join(self.root_dir, cat_id, model_id, 'rendering/renderings.txt')
		with open(img_names, 'r') as f:
			names = [line.strip() for line in f]
		with open(meta_path, 'r') as f:
			metas = [line.strip() for line in f]
		if len(names) != hyperparameters["num_rendering"] or len(metas) != hyperparameters["num_rendering"]:
			return False, result
		for i in range(hyperparameters["num_rendering"]):
			info = metas[i].split()
			if len(info) != 5:
				continue
			az, al = float(info[0]), float(info[1])
			if az > hyperparameters["azimuth_range"] and az < 360 - hyperparameters["azimuth_range"]:
				continue
			if al > hyperparameters["altitude_range"]:
				continue
			result.append(names[i])
		return True, result

	def get_model_names(self, class_root):
		model_names = [name for name in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, name))]
		result = sorted(model_names)
		if hyperparameters["max_model_per_class"] > 0 and len(result) > hyperparameters["max_model_per_class"]:
			result = result[:hyperparameters["max_model_per_class"]]
		return result


