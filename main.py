import os
import argparse
import sys
import numpy as np
from cbd.rank_a import compute_a
from cbd.factorization_j import fj


def get_a_list(pt, weight, model_name, model_dir):
    a = np.array([])
    btn = CBD_param['bottleneck_ratio']
    a_file_name = os.path.join(model_dir,'btn_'+str(btn)+'_a_list.npy')
    if os.path.isfile(a_file_name):
        print ("a_list exists when bottleneck is"+str(btn))
        with open(a_file_name, 'rb') as f:
            a = np.load(f)
    else:
        print ("Computing a...")
        a = compute_a(prototxt=pt, source=weight, model=model_name, source_path=model_dir, bottleneck=btn)
    return a

def factorization(a_list, pt, weight, model_name, model_dir):
    j = CBD_param['j']
    btn = CBD_param['bottleneck_ratio']
    fj(pt, weight, model_name, model_dir, j, btn, a_list)
    return

def cb_decomp(pt, weight, CBD_param):
    model_name = pt.split('/')[-1].split('.')[0]
    model_name1 = weight.split('/')[-1].split('.')[0]
    assert model_name == model_name1, "Please name prototxt and weight file by the same way."
    model_path='models/'
    model_dir = model_path+model_name
    j_dir = os.path.join(model_dir, 'j_model')
    for d in [model_dir, j_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    a_list = get_a_list(pt=pt, weight=weight, model_name=model_name, model_dir=model_dir)
    result = factorization(a_list=a_list, pt=pt, weight=weight, model_name=model_name, model_dir=model_dir)

def parse_args():
    parser = argparse.ArgumentParser("decouple CNN")
    # args for CBDNet
    parser.add_argument('-bottleneck', dest='btn', help='bottleneck ratio', default=None, type=float)
    parser.add_argument('-j', dest='j', help='total number of non-compressed binary bits except the sign channel', default=None, type=int)
    parser.add_argument('-model', dest='model', help='caffe prototxt file path', default=None, type=str)
    parser.add_argument('-weight', dest='weight', help='caffemodel file path', default=None, type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    caffe_path = '/home/path-to-caffe/'
    sys.path.insert(0, caffe_path+"python")
    
    args = parse_args()
    CBD_param = edict()
    CBD_param['bottleneck_ratio'] = args.btn
    CBD_param['j'] = args.j
    cb_decomp(pt=args.model,weight=args.weight,CBD_param=CBD_param)
