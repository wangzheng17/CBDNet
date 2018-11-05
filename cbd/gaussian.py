import numpy as np

def dist(matA, matB):
    newmatA = np.mod((matA + matB), 2)
    tmp = np.sum(newmatA)
    return tmp

def pivot_matrix(M, lm, j):
    m = max(M.shape[0], M.shape[1])
    n = min(M.shape[0], M.shape[1])
    idm = np.identity(m)
    row = max(range(j, m), key=lambda i: M[i][j])
    exchange = False
    if j != row:
        exchange = True
        tmp = M[row].copy()
        M[row] = M[j]
        M[j] = tmp
        tmp = lm[:,row].copy()
        lm[:, row] = lm[:, j]
        lm[:, j] = tmp
    return exchange

def exchange_zero_rows(M, lm):
    r_sum = np.sum(M, axis=1)
    zero_row = np.where(r_sum==0)[0]
    non_zero_row = np.where(r_sum!=0)[0]
    non_zero_row = np.flip(non_zero_row, axis=0)
    for r in non_zero_row:
        if len(zero_row)>0 and r > zero_row[0]:
            j = zero_row[0]
            tmp = M[r].copy()
            M[r] = M[j]
            M[j] = tmp
            tmp = lm[:,r].copy()
            lm[:, r] = lm[:, j]
            lm[:, j] = tmp
            zero_row=np.delete(zero_row,0)

def row_trans_matrix(h, i_idx, j):
    idm = np.identity(h)
    invert = np.identity(h)
    for i in i_idx:
        idm[i, j] = -1
        invert[i, j] = 1
    return idm, invert

def bindot(m1,m2):
    return np.mod(np.dot(m1, m2), 2)

def gauss_elimi(M):
    M = M.astype(np.int8)
    h = M.shape[0]
    w = M.shape[1]
    transpose = False
    if h < w:
        M = M.T
        transpose = True
    h = M.shape[0]
    w = M.shape[1]
    m = M.copy()
    lm = np.identity(h)
    mark = 0
    for j in range(w):
        exchange = pivot_matrix(M, lm,j)
        if exchange:
            pass
        row_idx = np.where(M[:,j]==1)[0]
        if row_idx.shape[0] > 1 and j in row_idx:
            p_idx = np.where(row_idx==j)[0][0]
            row_idx = row_idx[p_idx+1:]
            M[row_idx,:] = np.mod(M[row_idx,:]-M[j,:], 2)
            for ri in row_idx:
                lm[:,j] = np.mod(lm[:,ri]+lm[:,j], 2)
    return M, lm, transpose
    
def gauss_rank(M):
    up, inv, trans = gauss_elimi(M)
    r_sum = np.sum(up, axis=1) 
    return np.sum(r_sum!=0)

