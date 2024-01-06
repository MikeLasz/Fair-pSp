from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to pSp model checkpoint')
		self.parser.add_argument('--data_path', type=str, default='gt_images', help='Path to directory of images to evaluate')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')
		#self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')
		self.parser.add_argument('--resize_outputs', type=int, help='Whether to resize outputs to given size x size')

		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		# arguments for style-mixing script
		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')
		self.parser.add_argument('--n_outputs_to_generate', type=int, default=5, help='Number of outputs to generate per input image.')
		self.parser.add_argument('--mix_alpha', type=float, default=None, help='Alpha value for style-mixing')
		self.parser.add_argument('--latent_mask', type=str, default=None, help='Comma-separated list of latents to perform style-mixing with')

		# arguments for super-resolution
		self.parser.add_argument('--resize_factors', type=str, default=None,
		                         help='Downsampling factor for super-res (should be a single value for inference).')
  
		# parameters from https://bmvc2022.mpi-inf.mpg.de/0182.pdf 
		self.parser.add_argument('--use_augmentation', action='store_true', help='If activatated, we use the augmentations proposed by Tanjim for training.')
		self.parser.add_argument('--use_resampling', action='store_true', help='If activated, make use weighted training sampler to sample each label at the same frequency.')
		self.parser.add_argument('--use_mlp', action='store_true', help="Whether to 1. normalize Latents and 2. Apply an MLP afterwards before passing them to the Decoder.")
		self.parser.add_argument('--use_predicted_labels', action='store_true', help="Whether to use the predicted labels instead of the real labels for computing the Contrastive and the Cross-Entropy loss.")

		self.parser.add_argument('--rerun', type=int, default=1, help="Upsample an Image multiple times.")

	def parse(self):
		opts = self.parser.parse_args()
		return opts