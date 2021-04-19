import torch.nn as nn
import torch
import torch.nn.functional as F

class Discriminator(nn.Module):
	def __init__(self, in_channels, n_classes, base_n_filter = 8):
		super(Discriminator, self).__init__()
		self.in_channels = in_channels
		self.n_classes = n_classes
		self.base_n_filter = base_n_filter

		self.lrelu = nn.LeakyReLU()
		self.dropout3d = nn.Dropout3d(p=0.6)
		self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
		self.softmax = nn.Softmax(dim=1)

		# Level 1 context pathway
		self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
		self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
		self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

		# Level 2 context pathway
		self.conv3d_c4 = nn.Conv3d(self.base_n_filter, self.base_n_filter*8, kernel_size=3, stride=8, padding=1, bias=False)
		self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

		self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
		self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

		self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

		self.fc1 = nn.Linear(4096, n_classes)
        
	def conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def norm_lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def lrelu_conv(self, feat_in, feat_out):
		return nn.Sequential(
			nn.LeakyReLU(),
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

	def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
		return nn.Sequential(
			nn.InstanceNorm3d(feat_in),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2, mode='nearest'),
			# should be feat_in*2 or feat_in
			nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
			nn.InstanceNorm3d(feat_out),
			nn.LeakyReLU())

	def forward(self, x):
		#  Level 1 context pathway
		out = self.conv3d_c1_1(x)
		residual_1 = out
		out = self.lrelu(out)
		out = self.conv3d_c1_2(out)
		out = self.dropout3d(out)
		out = self.lrelu_conv_c1(out)
		# Element Wise Summation
		out += residual_1
		context_1 = self.lrelu(out)
		out = self.inorm3d_c1(out)
		out = self.lrelu(out)

		# Level 2 context pathway
		out = self.conv3d_c4(out)
		residual_4 = out
		out = self.norm_lrelu_conv_c4(out)
		out = self.dropout3d(out)
		out = self.norm_lrelu_conv_c4(out)
		out += residual_4
		out = self.inorm3d_c4(out)
		out = self.lrelu(out)
		context_4 = out

		out = self.conv3d_l0(out)
		out = self.inorm3d_l0(out)
		out = self.lrelu(out)
		out = self.ds2_1x1_conv3d(out)
		out = out.permute(0, 2, 3, 4, 1).contiguous().view(out.shape[0], -1)
		out = self.fc1(out)

		return out