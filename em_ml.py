import numpy as np

# A coin-flipping experiment
class coin(object):
    def __init__(self, p):
        self.o = np.array([1, 0], dtype='int')
        self.p = np.array([p, 1-p])
    
    def filp(self):
        return np.random.choice(self.o, p=self.p)


class coin_test(object):
    def __init__(self, cl, cp):
        self.cl = cl
        self.cp = cp

    def getdata(self, n):
        data = []
        for _ in range(n):
            i_c = np.random.choice(self.cl, p=self.cp)
            data.append(i_c.filp())
        return np.array(data, dtype='int')

    def grouped_test(self, n, m):
        data = []
        for _ in range(n):
            i_c = np.random.choice(self.cl, p=self.cp)
            data.append([i_c.filp() for i in range(m)])
        return np.array(data, dtype='int')

# em for mixed coins
coin1 = coin(0.5)
coin2 = coin(0.7)
coin_p = [0.5, 0.5]

test1 = coin_test([coin1, coin2], coin_p)
data1 = test1.grouped_test(200, 50)

# scheme 1

# MLE for coin
from scipy.stats import binom

def coin_p(ob, coin):
    ob_h = np.sum(ob)
    ob_t = len(ob)-ob_h
    return binom.pmf(ob_h, len(ob), coin.p[0])

def mle_p(ds):
    _all = 0
    _a_head = 0
    for i in ds:
        _all += len(i)
        _a_head += np.sum(i)
    if _all == 0:
        return 0.001
    else:
        return _a_head/_all
    

# init 
p1 = 0.2
p2 = 0.6
max_iteration = 1000

for i in range(max_iteration):
    c1 = coin(p1)
    c2 = coin(p2)
    z1 = []
    z2 = []
    for o in data1:
        if coin_p(o, c1) > coin_p(o, c2):
            z1.append(o)
        else:
            z2.append(o)
    # update p
    _p1 = mle_p(z1)
    _p2 = mle_p(z2)
    if _p1==p1 and _p2==p2:
        print(i)
        break
    else:
        p1 = _p1
        p2 = _p2
    
print(p1, p2)

# scheme 2
# em ml
def weight_c(ob, c1, c2):
    p1 = coin_p(ob, c1)
    p2 = coin_p(ob, c2)
    z1 = p1/(p1+p2)
    z2 = p2/(p1+p2)
    return (z1, z2)

def weight_emc(data, z):
    h, w = data.shape
    data_sum = np.sum(data, axis=1).reshape((h,1))
    data_all = (np.ones_like(data_sum)*w).reshape((h,1))
    p = np.sum(z * data_sum / data_all, axis=0) / np.sum(z, axis=0)
    return p
     

# init 
p1 = 0.7
p2 = 0.8
max_iteration = 1000
z = np.zeros((data1.shape[0], 2))

for i in range(max_iteration):
    c1 = coin(p1)
    c2 = coin(p2)
    for idx, o in enumerate(data1):
        z[idx] = weight_c(o, c1, c2)
    # update p
    _p1, _p2 = weight_emc(data1, z)
    if _p1==p1 and _p2==p2:
        print(i)
        break
    else:
        if i % 100 == 0:
            print(_p1, _p2)
        p1 = _p1
        p2 = _p2
    
print(p1, p2)


coin1 = coin(0.7)
coin2 = coin(0.7)

test2 = coin_test([coin1, coin2], [0.3, 0.7])
data1 = test2.grouped_test(200, 5)



# init 
p1 = 0.1
p2 = 0.8
max_iteration = 1000
z = np.zeros((data1.shape[0], 2))

for i in range(max_iteration):
    c1 = coin(p1)
    c2 = coin(p2)
    for idx, o in enumerate(data1):
        z[idx] = weight_c(o, c1, c2)
    # update p
    _p1, _p2 = weight_emc(data1, z)
    if _p1==p1 and _p2==p2:
        print(i)
        break
    else:
        if i % 100 == 0:
            print(_p1, _p2)
        p1 = _p1
        p2 = _p2
    
print(p1, p2)
