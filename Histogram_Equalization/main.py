import copy
import os
import matplotlib.pyplot as plt
import numpy as np

L = 256
OutputDir = "output"
ALL_FILES = []
CURRENT_FILE = ""
IF_SHOW = False

def show_hist(pic1: np.array, pic2: np.array):
	global CURRENT_FILE

	fig, ax = plt.subplots(nrows=1, ncols=2)
	ax1:plt.Axes = ax[0]
	ax2:plt.Axes = ax[1]

	ax1.hist(pic1.flatten(), bins=L, color='b', label='Before')
	ax1.set_xlim((0, 255))
	ax1.set_xticks(np.linspace(start=0, stop=255, num=5, dtype=int))
	ax1.legend()

	ax2.hist(pic2.flatten(), bins=L, color='r', label='After')
	ax2.set_xlim((0, 255))
	ax2.set_xticks(np.linspace(start=0, stop=255, num=5, dtype=int))
	ax2.legend()

	if IF_SHOW:
		plt.show()
	else:
		plt.savefig(os.path.join(OutputDir, "Hist_" + CURRENT_FILE+".png"))

def show_pic(pic1: np.array, pic2: np.array):
	global CURRENT_FILE

	fig, ax = plt.subplots(nrows=1, ncols=2)
	ax1:plt.Axes = ax[0]
	ax2:plt.Axes = ax[1]

	ax1.imshow(pic1, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
	ax1.set_yticks([])
	ax1.set_xticks([])
	ax1.set_xlabel('Before')

	ax2.imshow(pic2, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
	ax2.set_xticks([])
	ax2.set_yticks([])
	ax2.set_xlabel('After')

	if IF_SHOW:
		plt.show()
	else:
		plt.savefig(os.path.join(OutputDir, "Pic_" + CURRENT_FILE+".png"))

def get_pic_feature(hist: np.array):
	return np.mean(hist), np.std(hist)

def main():
	global CURRENT_FILE

	_, _, ALL_FILES = list(os.walk("./data"))[0]
	print(ALL_FILES)

	os.makedirs(OutputDir, exist_ok=True)
	
	for file in ALL_FILES:
		CURRENT_FILE = os.path.basename(file)

		pic = plt.imread(os.path.join("data", file))
		H, W = pic.shape

		counts = np.bincount(pic.flatten(), minlength=L)
		accumulate_count = copy.deepcopy(counts)
		for i in range(1, len(counts)):
			accumulate_count[i] += accumulate_count[i-1]
		Transform = (L-1) / (H * W) * accumulate_count

		new_pic = copy.deepcopy(pic).flatten()
		for i in range(new_pic.size):
			new_pic[i] = Transform[new_pic[i]]
		new_pic = new_pic.reshape(pic.shape)

		show_pic(pic, new_pic)
		show_hist(pic, new_pic)

		pic_feature = get_pic_feature(pic)
		new_pic_feature = get_pic_feature(new_pic)

		print('\t-: ' + CURRENT_FILE+" Before: ", pic_feature)
		print('\t+: ' + CURRENT_FILE+" After: ", new_pic_feature)


if __name__ == "__main__":
	main()