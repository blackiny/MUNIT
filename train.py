"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images, Timer, write_image_display
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
# import tensorboardX
from torch.utils.tensorboard import SummaryWriter
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
if not os.path.exists(opts.output_path):
    os.makedirs(opts.output_path)
model_name = os.path.splitext(os.path.basename(opts.config))[0]
display_directory = os.path.join(opts.output_path, "logs")
if not os.path.exists(display_directory):
    os.makedirs(display_directory)
train_writer = SummaryWriter(log_dir = os.path.join(display_directory, model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
ep0, iterations = -1, 0
if opts.resume:
    ep0, iterations = trainer.resume(checkpoint_directory, hyperparameters=config)
ep0 += 1
print('start the training at epoch %d'%(ep0 + 1))
for ep in range(ep0, config['n_ep']):
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        # trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()
        # lr_update called after optimizer
        trainer.update_learning_rate()
        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print('total_it: %d (ep %d, it %d), gen lr %08f, dis lr %08f' % (
                iterations + 1, ep + 1, it + 1, trainer.gen_opt.param_groups[0]['lr'], trainer.dis_opt.param_groups[0]['lr']))
            # print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        if (iterations + 1) % config['image_display_freq'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(images_a, images_b)
            write_image_display(image_outputs, images_a.size(0), iterations, train_writer)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
    
    # Save network weights
    if (ep + 1) % config['snapshot_save_freq'] == 0:
        trainer.save(checkpoint_directory, ep, iterations)

    # Write images
    if (ep + 1) % config['image_save_freq'] == 0:
        with torch.no_grad():
            test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
            train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
        write_2images(test_image_outputs, display_size, image_directory, 'test_%05d' % (ep + 1))
        write_2images(train_image_outputs, display_size, image_directory, 'train_%05d' % (ep + 1))

