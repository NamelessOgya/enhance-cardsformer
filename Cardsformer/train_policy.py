from Algo.arguments import parser
from Algo.dmc import train
import os


if __name__ == "__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
	
	flags = parser.parse_args()
	print("==================")
	print(flags)
	print("==================")

	model_save_dir = os.path.expanduser('%s/%s' %
                           (flags.savedir, flags.xpid))
	
	print(model_save_dir)
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	
	
	train(flags)
