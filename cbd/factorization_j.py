import os
import numpy as np
from cbd.gaussian import gauss_elimi, gauss_rank, bindot, exchange_zero_rows
import caffe
def dist(matA, matB):
    newmatA = np.mod((matA - matB), 2)
    tmp = np.count_nonzero(newmatA)
    return tmp

#model = "SSD"
#model = "MobileNetSSD"
#model = "resnet-152"
#model = "DenseNet_121"
#model = "resnet-18"
#model = "VGG_16"

def fj(prototxt, source, model, source_path, j, bottleneck, a):
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
    prototxt = os.path.join(source_path, model+".prototxt")
    source   = os.path.join(source_path, model+".caffemodel")
    qtarget = os.path.join(source_path, "j_model/", model + "_j_" + str(j+1) + "_btn_" + str(bottleneck) + ".caffemodel")
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, source, caffe.TEST)
    layers = net.params.keys()
    n = 0
    all_d = 0
    all_d2 = 0
    bpl = []
    num_bin_param = 0.
    num_param = 0.
    over_size = 0.
    J = j
    a_s_arr = []
    for idx, layer in enumerate(layers):
        print (layer)
        if all([not ext_layer[e] in layer for e in range(len(ext_layer))]):
            w = net.params[layer][0].data
            wMax = np.max(np.abs(w))
            r = w/wMax # normalize
            sign = np.sign(r)
            r = np.absolute(r)
            q = r.copy()
            if (model=='VGG_16' or model=='resnet-18') and 'fc' in layer:
                height = w.shape[0]
                width = w.shape[1]
            else:
                height = w.shape[0] * w.shape[2]
                width = w.shape[1] * w.shape[3]
            num_param += height*width
            a_s = np.floor(np.log2(np.max(r/a[n])))
            print (a_s)
            a_s_arr.append(a_s)
            a_i = a_s-J+1
            D = 2**(a_i-1)/(1-2**(a_i-1)) * a[n]
            ### code improve, better approximate a_s with D
            #a_s = np.floor(np.log2(np.max((r+D)/(a[n]+D))))
            #a_i = a_s-J+1
            ###
            print (n, w.shape, np.max(np.max((r+D)/(a[n]+D))), a[n])
            num_bin_param_lay = 0.
            a_s = int(a_s)
            for i in range(a_s, a_s-J, -1):
                r_idx = r > 2**(i)*(a[n]+D)-D
                print ("==========")
                if i>=0:
                    # factorization
                    r_tmp_idx = r_idx.reshape([height,width]).copy()
                    up, invert, trans = gauss_elimi(r_tmp_idx.copy())
                    r_sum = np.sum(up,axis = 1)
                    nonzero_row = np.where(r_sum != 0)[0]
                    zero_row = np.where(r_sum == 0)[0]
                    rr = len(nonzero_row)
                    num_bin_param_lay += rr*(height+width)
                    print ("up", up.shape, min(up.shape)*bottleneck, " -> ", rr)
                    print ("n, i, btn, o_size, c_size")
                    print (n, i, rr, height*width, rr*(height+width))
                    """
                    ## following codes verifies A = BC
                    # A=BC
                    exchange_zero_rows(up, invert) # make first rr rows all non-zero
                    C = up[:rr, :]
                    tmpI = np.zeros([up.shape[0], rr])
                    for ri in range(rr):
                        tmpI[ri,ri]=1
                    B = bindot(invert, tmpI)
                    r_new_idx = bindot(B, C)
                    if trans:
                        r_new_idx = r_new_idx.T
                    print 'dist ', dist(r_idx, r_new_idx.reshape(r_idx.shape))
                    """
                r = r - (r_idx)*((a[n]+D)*2**i)
            if num_bin_param_lay > height*width: # do not compress
                over_size += 1
                num_bin_param += height*width*(J+1) # non-compressed value and sign
                delta = 2**(-J-1)/(1-2**(-J-1)) # 
                q = np.round(q/delta)*delta
                print ("oversize: ", num_bin_param_lay, height*width)
                
            else: # compress
                num_bin_param += height*width*(abs(a_i)+1) # non-compressed value and sign
                num_bin_param += num_bin_param_lay # compressed part
                print ("test_w", np.max(r), np.min(r), D)
                q =  q - r
            q = sign * q
            q = q*wMax
            np.copyto(net.params[layer][0].data,q)
            n += 1
    net.save(qtarget)
    print (a)
    print ('btn', bottleneck, 'a_i', a_i, 'J', J+1)
    print ("Originial size:", num_param*32/8/1024/1024, "MB")
    print ("Compressed size:", num_bin_param/8/1024/1024, "MB")
    print ("bit-rate", num_bin_param/num_param)
    print ("Oversize", over_size, "/", n)
    with open(os.path.join(source_path, 'j_model/', model+'_result_btn_'+str(bottleneck)+'_j_'+str(J+1)+'.txt'), 'w') as f:
        f.write("Originial size: "+str(num_param*32/8/1024/1024)+"MB"+'\n')
        f.write("Compressed size:"+str(num_bin_param/8/1024/1024)+"MB"+'\n')
        f.write("bit-rate: "+str(num_bin_param/num_param)+'\n')
        f.write("Oversize: "+str(over_size)+"/"+str(n)+'\n')
        f.write("Model path: "+qtarget+'\n')
        f.write("Compressed Channel: "+str(a_s_arr)+'\n')
