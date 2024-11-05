import numpy as np


w_init = np.array(
    [0.5,0,-1,1]
) # init weight

w = [w_init]

u = 0.5 #learning rate

d = np.array([
    [1],
    [1]
]) #output

epoch = 10

x = np.array([
    [-1,0,-2,1],
    [-1,-0.5,1.5,0]
]) #input

def main():
    for i in range(epoch):
        w_temp = w[i].copy()
        # print(w_temp)
        w_index = i%len(x)
        dot_v = np.sign(np.dot(w_temp.T,x[w_index])) 
        print("Wt:",w_temp.T)
        print("x :",x[w_index])
        print("dv:",dot_v)
        w_temp += (0.5*(d[w_index]-dot_v)*x[w_index])
        print(w_temp)
        print("----------")
        w.append(w_temp)
    print("|---------|")
    print(w[-1])
    print("|---------|")
    # print(np.linalg.pinv(x))
    print(np.round(np.dot(np.linalg.pinv(x),d),4).T)
if __name__ == '__main__':
    main()