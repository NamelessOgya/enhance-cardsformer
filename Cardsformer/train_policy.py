import os
# 環境変数を設定
os.environ["PYTHONNET_RUNTIME"] = "coreclr"

# 設定した環境変数を取得
print(os.environ["PYTHONNET_RUNTIME"])

from Algo.arguments import parser
from Algo.dmc import train



if __name__ == "__main__":

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	
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
