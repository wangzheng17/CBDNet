import os
import numpy as np
#import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank 
from cbd.gaussian import gauss_rank
import caffe

def binrank(aa):
    aa_height = aa.shape[0]
    aa_width = aa.shape[1]
    mark = np.zeros(aa_height)

    for i in range(0,aa_width):
        indexj = np.where(aa[:,i] == 1)
        if (indexj[0].shape[0] != 0):# if some elements equals 1
            j = indexj[0][0]
            mark[j] = 1
            for x in range(0,aa_width):
                if ((aa[j,x] == 1) and (i != x)):
                    aa[:,x] = np.mod(aa[:,x] + aa[:,i],2)
    return np.sum(mark)

#model = "SSD"
#model = "MobileNetSSD"
#model = "resnet-152"
#model = "DenseNet_121"
#model = "resnet-18"
#model = "VGG_16"
#bottleneck = 0.5

def compute_a(prototxt, source, model, source_path, bottleneck):
    if model == "SSD":
        ext_layer = ['bn', 'scale', 'mbox', 'norm']
    if model == "MobileNetSSD":
        ext_layer = ['bn', 'scale', 'mbox']
    if model == "resnet-152":
        ext_layer = ['bn', 'scale', 'fc1000']
    if model == "resnet-18":
        ext_layer = ['bn', 'scale']
    elif model == "VGG_16":
        ext_layer = []
    elif model == "DenseNet_121":
        ext_layer = ['bn', 'scale']
    else:
        raise Exception('Please use model name compatiable with that in line 33-43, rank_a.py')
    prototxt = os.path.join(source_path, model+".prototxt")
    source   = os.path.join(source_path, model+".caffemodel")

    #caffe.set_mode_cpu()
    net = caffe.Net(prototxt, source, caffe.TEST)
    layers = net.params.keys()

    detail = False
    a = []
    n = 0
    for idx, layer in enumerate(layers):
        if all([not ext_layer[e] in layer for e in range(len(ext_layer))]):
            
            w = net.params[layer][0].data
            if model=='MobileNetSSD' and w.shape[-1] == 1 and w.shape[-2] == 1:
                continue
            print (w.shape, layer)
            n+=1
            #continue
            wMax = np.max(np.abs(w))
            r = w/wMax # normalize
            if (model=='VGG_16' or model=='resnet-18') and 'fc' in layer:
                height = w.shape[0]
                width = w.shape[1]
            else:
                height = w.shape[0] * w.shape[2] 
                width = w.shape[1] * w.shape[3]

            bound = 0
            maxlength = 0
            minlength = 0
            if height < width:
                maxlength = width
                minlength = height
            else:
                maxlength = height
                minlength = width
            bound = int(minlength * bottleneck)
            if bound == 0:
                raise
            print (bound)
            w_shape = height*width
            w_one = np.reshape(r, height * width)        
            w_copy = np.abs(np.reshape(r,[height, width]))
            w_one = np.absolute(w_one)
            w_sort = np.sort(w_one)
            w_sort = w_sort[::-1]
            r_tmp = np.reshape(np.abs(r), (height, width))
            #binary search
            min = 0
            max = int((minlength*maxlength) * bottleneck)
            #max = int((bound-1)*(bound-1))
            print (max)
            r_tmp_idx = r_tmp > w_sort[max]
            rank = gauss_rank(r_tmp_idx)
            if rank > bound:
                while True:
                    center = int((min + max)/2)
                    r_tmp_idx = r_tmp > w_sort[center]
                    rank = gauss_rank(r_tmp_idx)
                    print (min, max, center, rank)
                    if center != 0 and rank == 0:
                        min += 1
                        continue
                    if max <= min:
                        print ('min>=max, rank, bound, value')
                        print (min, max, rank, bound, w_sort[center])
                        a.append(w_sort[center])
                        break
                    if rank > bound:
                        max = center - 1
                    elif rank < bound:
                        min = center + 1
                    elif rank == bound:
                        #my_rank = gauss_rank(w_copy>w_sort[center])
                        #print ("my_rank", my_rank, w_sort[center])
                        print ('-----id, rank, value-----')
                        print (center, rank, w_sort[center])
                        a.append(w_sort[center])
                        if detail:
                            print ('===============>')
                            for i in range(0, 20):
                                r_tmp_idx = r_tmp > w_sort[center+i]
                                rank = gauss_rank(r_tmp_idx)
                                print ('id, rank, value',center+i, rank, w_sort[center+i])
                        break
            else:
                print ('good matrix')
                print (rank, w_sort[max])
                a.append(w_sort[max])
    a_tmp = np.asarray(a)
    with open(os.path.join(source_path, 'btn_'+str(bottleneck)+'_a_list.npy'), 'wb') as f:
       np.save(f, a_tmp)
    print ('btn', bottleneck)
    return a_tmp
