from copy import deepcopy
import os
from matplotlib.image import imsave
import matplotlib.pyplot as plt
import numpy as np
import cv2

OutputDir = "output"
ALL_FILES = []
CURRENT_FILE = ""
IF_SHOW = False
Suffix = ".png"
SegmentMethod = ['otsu', 'growing'][1]

growing_config = {
	'1': {'seeds': [[50, 60]], 'c': 0.2},
	'2': {'seeds': [[30, 30]], 'c': 0.2},
	'3': {'seeds': [[80, 80], [140, 60], [42, 112], [78, 155]], 'c': 0.1},
	'4': {'seeds': [[50, 60]], 'c': 0.1},
	'5': {'seeds': [[60, 60]], 'c': 0.2},
	'6': {'seeds': [[20, 40]], 'c': 0.4},
	'b': {'seeds': [[60, 110], [250, 180]], 'c': 0.3},
	'lena': {'seeds': [[100, 450], [250, 460], [50, 100], [150, 50], [200, 10], [300, 500], [20, 460], [50, 440]], 'c': 0.15},
	'objs': {'seeds': [[60, 60], [80, 110]], 'c': 0.2},
}

def show_pic(pic1: np.array, str1='', scale=True):
	global CURRENT_FILE
	pic1 = pic1.squeeze()

	filename = os.path.join(OutputDir, str1+'_'+CURRENT_FILE+Suffix)

	if scale:
		imsave(filename, pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
	else:
		imsave(filename, pic1, cmap=plt.get_cmap('gray'))

	if IF_SHOW:
		if scale:
			plt.imshow(pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
		else:
			plt.imshow(pic1, cmap=plt.get_cmap('gray'))
		plt.xticks([])
		plt.yticks([])
		plt.title(CURRENT_FILE)
		plt.show()

def show_with_contour(pic: np.array, contour: np.array, seeds: np.array = None, str1=''):
	global CURRENT_FILE
	pic = pic.squeeze()

	filename = os.path.join(OutputDir, 'c_'+str1+'_'+CURRENT_FILE+Suffix)

	plt.imshow(pic, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
	plt.contour(contour, colors='red')
	if not seeds is None:
		plt.scatter(seeds[:, 1], seeds[:, 0], color='green', marker='x', linewidths=2)
	plt.xticks([])
	plt.yticks([])
	plt.savefig(filename, bbox_inches='tight', pad_inches=-0.1)
	plt.close()

	filename = os.path.join(OutputDir, 'cf_'+str1+'_'+CURRENT_FILE+Suffix)

	pic = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
	contour = np.repeat(contour[:,:,np.newaxis], 3, axis=-1)
	pic = np.where(contour == 0, pic, np.array([16, 200, 241]))
	plt.imshow(pic, vmin=0,vmax=255)
	# plt.contour(contour, colors='red')

	if not seeds is None:
		plt.scatter(seeds[:, 1], seeds[:, 0], color='green', marker='x', linewidths=2)
	plt.xticks([])
	plt.yticks([])
	plt.savefig(filename, bbox_inches='tight', pad_inches=-0.1)
	plt.close()

	if IF_SHOW:
		plt.imshow(pic, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
		plt.xticks([])
		plt.yticks([])
		plt.title(CURRENT_FILE)
		plt.show()


def otsu(pic: np.array) -> np.array:
	Ls = np.array(range(256))
	cnt = np.bincount(pic.flatten(), minlength=256)
	assert(sum(cnt) == pic.size)
	p_cnt = cnt / pic.size

	m_G = np.sum(Ls * p_cnt)

	all_Sigma = np.zeros(256)
	for k in range(1, 256):
		L1, L2 = Ls[:k], Ls[k:]
		p_cnt1, p_cnt2 = p_cnt[:k], p_cnt[k:]
		P1, P2 = np.sum(p_cnt1), np.sum(p_cnt2)
		if P1 == 0 or P2 == 0:
			continue
		m1 = np.sum(L1 * p_cnt1) / P1
		m2 = np.sum(L2 * p_cnt2) / P2

		sigma_B = P1 * ((m1 - m_G) ** 2) + P2 * ((m2 - m_G) ** 2)
		all_Sigma[k] = sigma_B
		
	k = all_Sigma.argmax()

	new_pic = np.where(pic < k, 0, 1)
	return new_pic


def growing(pic: np.array, seeds: np.array, c=0.2) -> np.array:
	std = np.std(pic.flatten())
	d = [[-1, 0], [0, 1], [1, 0], [0, -1]]
	H, W = pic.shape

	ret_pic = np.zeros_like(pic)
	st = np.zeros_like(pic)
	for seed in seeds:
		st[seed[0], seed[1]] = 1
	while len(seeds) > 0:
		h, w = seeds[0]
		ret_pic[h, w] = 1
		seeds = seeds[1:]
		intensity = pic[h, w]
		for dh, dw in d:
			nh, nw = h + dh, w + dw
			if not ( nh >= 0 and nw >= 0 and nh < H and nw < W ):
				continue
			if not st[nh, nw] == 0:
				continue
			n_intensity = pic[nh, nw]
			# print ('\t[{}, {}] = {}'.format(nh, nw, n_intensity))
			if np.abs(intensity - n_intensity) < c * std:
				st[nh, nw] = 1
				seeds = np.concatenate([seeds, np.array([[nh, nw]])], axis=0)
	return ret_pic


def main():
	global CURRENT_FILE

	_, _, ALL_FILES = list(os.walk("./data"))[0]
	print('\033[1;32m', ALL_FILES, '\033[0m')

	for file in ALL_FILES:
		CURRENT_FILE = os.path.splitext(os.path.basename(file))[0]
		assert(CURRENT_FILE in growing_config)
		print('\033[31m', 'Processing "' + CURRENT_FILE + '"', '\033[0m')

		pic = plt.imread(os.path.join("data", file))

		if SegmentMethod == 'otsu':
			new_pic = otsu(pic)
			show_with_contour(pic, new_pic, None, 'Otsu1')
			show_with_contour(pic, 1-new_pic, None, 'Otsu2')
		else:
			seeds = np.array(growing_config[CURRENT_FILE]['seeds'])
			c = growing_config[CURRENT_FILE]['c']
			new_pic = growing(pic.astype(np.int32), seeds, c)

			show_with_contour(pic, new_pic, seeds, 'Growing1')
			show_with_contour(pic, 1-new_pic, seeds, 'Growing2')

if __name__ == "__main__":
	main()
