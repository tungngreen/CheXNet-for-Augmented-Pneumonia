import argparse
import time

#args = argparse.ArgumentParser('CheXNet')

args = argparse.ArgumentParser(fromfile_prefix_chars='@')

#paths /home/tung/CS570/team_prj/chexnet-edited/config.py
args.add_argument('--train_dir', type = str, default = '/datasets/chest_xray/train')
args.add_argument('--test_dir', type = str, default = '/datasets/chest_xray/test')
args.add_argument('--augmented_dir', type = str, default = None)
args.add_argument('--augmented_set', type = str, default = '')

#training schemes
args.add_argument('--real_train_portion', type = float, default = 1)
args.add_argument('--real_test_portion', type = float, default = 1)
args.add_argument('--aug_train_portion', type = float, default = 0)

#general settings
args.add_argument('--use_cuda', type = bool, default = True)
args.add_argument('--n_gpu', type = int, default = 1)
args.add_argument('--gpu_id', type = str, default = '2')
args.add_argument('--random_seed', type = int, default = int(time.time()))
args.add_argument('--train', type = bool, default = True)
args.add_argument('--test', type = bool, default = True)
args.add_argument('--test_after_training', type = bool, default = True)
args.add_argument('--testing_save_dir', type = bool, default = '')

#architecture
args.add_argument('--dense_net', type = str, default = '121')
args.add_argument('--is_trained', type = bool, default = False)
args.add_argument('--n_class', type = int, default = 2)
args.add_argument('--optimizer', type = str, default = 'Adam')

#input settings
args.add_argument('--image_size', type = int, default = 256)
args.add_argument('--crop_size', type = int, default = 224)

#training parameters
args.add_argument('--batch_size', type = int, default = 16)
args.add_argument('--n_epoch', type = int, default = 100)
args.add_argument('--lr', type = float, default = 0.001)
args.add_argument('--parallel_training', type = bool, default = True)
args.add_argument('--load_ckpt', type = bool, default = False)
args.add_argument('--ckpt_path', type = str, default = '')
#args.add_argument()


#config = parser.parse_known_args()

config, _ = args.parse_known_args()
#print(args.parse_args(['--augmented_dir']))