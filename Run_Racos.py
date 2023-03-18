from __future__ import print_function
import argparse

from Components import Instance
from Racos import *
from utils import *

import os
import time

from keras.models import Model

# read the parameter
parser = argparse.ArgumentParser(
    description='Main function for adverary input generation using Racos in ImageNet dataset')
# the parameters about racos
parser.add_argument('-iter', '--iterations', help="number of iterations of sampling",default=20000, type=int)
parser.add_argument('-sam', '--samples', help="number of samples in every iteration",default=3, type=int)
parser.add_argument('-pn', '--positive_num', help="the set size of PopPop", default=2, type=int)
parser.add_argument('-ub', '--uncertain_bits', help="the dimension size that is sampled randomly", 
                    default=10, type=int)
# the parameters about attack
parser.add_argument('-tm', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2, 3], default=3, type=int)
parser.add_argument('-mp', '--max_perturbation', help="the max perturbation in every pixel", 
                    default=10.0, type=float)
parser.add_argument('--target', action="store_true", default=False)
parser.add_argument('-tl', '--target_index', help="the ith target label", default=10, type=int)
parser.add_argument('-to', '--time_out', help="the limit of time about one image", default=9999999, type=int)
parser.add_argument('-ip', '--input_path', help="the path of imgs under attack", default="./100/", type=str)
parser.add_argument('-ie', '--image_ext', help="the ext of imgs under attack", default='jpg', type=str)
parser.add_argument('-se', '--seeds', help="number of seeds of input", default=100, type=int)
# parser.add_argument('-ih', '--is_hard', help="option of easy, hard or all", choices=["eazy", "hard", "all"], default="all", type=str)
parser.add_argument('-rs', '--resize', help="resize", default=100, type=int)
parser.add_argument('-rm', '--resize_mode', help="interpolation:nearest or bilinear", choices=["nearest", "bilinear"], default="bilinear", type=str)

args = parser.parse_args()

# check log dir exists
if not os.path.exists("./log"):
    os.mkdir("./log")

if True:
    model, img_rows, img_cols, img_rgb = load_model(args.target_model)
    # dimsize = args.resize or img_rows
    dimsize = args.resize
    dim = initialize_Dim(dimsize*dimsize*img_rgb, args.max_perturbation, -args.max_perturbation)
    img_set = initialize_img_set(args.input_path, args.image_ext)
    if args.seeds > len(img_set):
        args.seeds = len(img_set)
    filename = time.strftime("%m_%d_%H_%M_%S") #record the result
    initialize_log_file(filename, args)
    root_path, success_path, fail_path, npy_path = mkdir_folders(filename)

    racos = RacosOptimization(dim, model, img_rows, img_cols, img_rgb, 
        dimsize=dimsize, resize_mode=args.resize_mode, ss=args.samples, mt=args.iterations, 
        pn=args.positive_num, ub=args.uncertain_bits, to=args.time_out)
    
    # # main loop
    for index in range(args.seeds):

        ori_input = img_to_input(img_set[index], img_rows)
        ori_pred = input_to_prediction(np.expand_dims(ori_input, axis=0), model)
        ori_label = np.argmax(ori_pred)
        print(decode_predictions(ori_pred,6))

        time_begin = time.time()
        if args.target:
            target_label = find_k_max(ori_pred, args.target_index)
            racos.Opt(ori_input, ori_label, target_label)
        else:
            racos.Opt(ori_input, ori_label)
        time_cost = time.time()-time_begin

        if not dimsize == img_rows:
            noise = noise_resize(racos.getOptimal(), dimsize, img_rows, img_rgb, args.max_perturbation, args.resize_mode)
        else:
            noise = np.array(racos.getOptimal().getFeatures()).reshape(img_rows, img_cols, img_rgb)
        gen_img = np.clip(np.around(ori_input+noise), 0, 255)

        f = open("./log/"+filename, 'a')
        if racos.getOptimal().getFitness() == -10000000:
            f.write(str(index)+'_success_'+img_set[index]+'_'+str(max_abs(racos.getOptimal().getFeatures()))+\
                    '_'+str(racos.getQuerys()) +'_'+str(time_cost)+ '\n')
            save_img(gen_img, npy_path+str(ori_label)+'.npy', success_path+str(ori_label)+'.png')
        else:
            f.write(str(index)+'_fail_'+img_set[index]+ '_' +str(racos.getOptimal().getFitness())+'_'+str(racos.getQuerys()) +'_'+str(time_cost)+ '\n')
            save_img(gen_img, npy_path+str(ori_label)+'.npy', fail_path+str(ori_label)+'.png')
        f.close()
        













