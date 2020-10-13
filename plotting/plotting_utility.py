import seaborn as sns
import matplotlib.pyplot as plt
TICKLABELSIZE = 14
LEGENDLABELSIZE = 14
LABELSIZE = 23


def histogram_plot(plotting_df, x, y, hue=None, colors=None, num_x=6):
	bar = sns.barplot(data=plotting_df, x=x, y=y, hue=hue)

	for i, thisbar in enumerate(bar.patches):
		# Set a different hatch for each bar
		if i < (1*num_x):
			thisbar.set_color(colors[0])
			thisbar.set_hatch('\\')
			thisbar.set_edgecolor('black')
		elif i < (2*num_x):
			thisbar.set_color(colors[1])
			thisbar.set_hatch('/')
			thisbar.set_edgecolor('black')
		elif i < (3*num_x):
			thisbar.set_color(colors[2])
			thisbar.set_hatch('+')
			thisbar.set_edgecolor('black')
		else:
			thisbar.set_color(colors[3])
			thisbar.set_hatch('o')
			thisbar.set_edgecolor('black')
	plt.axvline(x=2.5, linestyle="dashed", color="black")
	plt.ylim(0.0, 0.74)
	plt.ylabel(y, fontsize=LABELSIZE)
	plt.xlabel("", fontsize=LABELSIZE)
	plt.tick_params(axis="x", labelsize=TICKLABELSIZE)
	plt.tick_params(axis="y", labelsize=TICKLABELSIZE)
	# plt.legend(loc="upper left", bbox_to_anchor=(1.04,1), fontsize=LEGENDLABELSIZE)
	plt.legend(loc="best", fontsize=LEGENDLABELSIZE)
	# plt.legend("", frameon=False)
	# plt.savefig("/data/anirudh/output/train_model_debugging/figs/class/comprehensiveness.pdf", bbox_inches = 'tight', dpi=300)
	# plt.savefig("/data/anirudh/output/train_model_debugging/figs/class/comprehensiveness.png", bbox_inches = 'tight', dpi=300)
	return plt