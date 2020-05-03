import numpy as np
import scipy
from scipy import cluster
# VAE related:

import numpy as np

def intra_inter_class_distance(cla1, cla2, num, embedded = False, plot = True, metric = "euclidean"):      
    data1 = np.array(class_dic[cla1][:num]).reshape(-1, 28*28)
    data2 = np.array(class_dic[cla2][:num]).reshape(-1, 28*28)
    if embedded:
        _, data1,_= vae(torch.from_numpy(data1).float())
        _, data2,_= vae(torch.from_numpy(data2).float())
        data1 = data1.detach().numpy()
        data2 = data2.detach().numpy()
    dist1 = []
    dist2 = []
    inter_dist = []
    for i in range(len(data1)):
        for j in range(len(data1)):
            x1 = data1[i]
            x2 = data1[j]
            if metric == "euclidean":
                dist1.append(np.linalg.norm(x1 - x2))
            if metric == "cosine":
                dist1.append(scipy.spatial.distance.cosine(x1, x2))
    for i in range(len(data2)):
        for j in range(len(data2)):
            x1 = data2[i]
            x2 = data2[j]
            if metric == "euclidean":
                dist2.append(np.linalg.norm(x1 - x2))
            if metric == "cosine":
                dist2.append(scipy.spatial.distance.cosine(x1, x2))
    for i in range(len(data1)):
        for j in range(len(data2)):
            x1 = data1[i]
            x2 = data2[j]
            if metric == "euclidean":
                inter_dist.append(np.linalg.norm(x1 - x2))
            if metric == "cosine":
                inter_dist.append(scipy.spatial.distance.cosine(x1, x2))
    if plot:
        _,_,_ = plt.hist(dist1, np.arange(200) * 1.2 * np.max(dist1)//200, histtype="step", label = cla1 + " "+ metric)
        _,_,_ = plt.hist(dist2, np.arange(200) * 1.2 * np.max(dist1)//200, histtype="step", label = cla2 + " "+ metric)
        _,_,_ = plt.hist(inter_dist, np.arange(200) * 1.2 * np.max(dist1)//200, histtype="step", label = cla1 + "-" + cla2 + " "+ metric)
        plt.legend()
        plt.show()        
    return np.array(dist1), np.array(dist2), np.array(inter_dist)

def plot_two_hist(cla1, cla2, num, metric = "euclidean"):
    fig ,ax = plt.subplots(1,2, figsize = (15, 8))
    dist1, dist2, inter_dist = intra_inter_class_distance(cla1, cla2, num, False, False, metric)
    if metric == "euclidean":
        bins = np.arange(200) * 1.2 * np.max(dist1)//200
    if metric == "cosine":
        bins = np.arange(-100, 100) * 1.2 * np.max(dist1)/100
    ax[0].hist(dist1, bins, histtype="step", label = cla1 + " " + metric)
    ax[0].hist(dist2, bins, histtype="step", label = cla2 + " " + metric)
    ax[0].hist(inter_dist, bins, histtype="step", label = cla1 + "+" + cla2 + " " + metric)
    ax[0].legend()
    dist1, dist2, inter_dist = intra_inter_class_distance(cla1, cla2, num, True, False, metric)
    if metric == "euclidean":
        bins = np.arange(200) * 1.2 * np.max(dist1)//200
    if metric == "cosine":
        bins = np.arange(-100, 100) * 1.2 * np.max(dist1)/100
    ax[1].hist(dist1, bins, histtype="step", label = cla1 + " " + metric)
    ax[1].hist(dist2, bins, histtype="step", label = cla2 + " " + metric)
    ax[1].hist(inter_dist, bins, histtype="step", label = cla1 + "+" + cla2 + " " + metric)
    ax[1].legend()



class node(cluster.hierarchy.ClusterNode):
    def __init__(self, id, left=None, right=None, dist=0, count=1):
        #super(node, self).__init__(id, left=None, right=None, dist=0, count=1)
        self.id = id
        self.left=left
        self.right=right
        self.dist=dist
        self.count=count
        self.parent = None
        
def create_tree(root):
    if root is None:
        return None
    new_left = create_tree(root.left)
    new_right = create_tree(root.right)
    new_root = node(root.id, new_left, new_right, root.dist, root.count)
    return new_root

def create_par(root, par):
    if root is None:
        return
    root.parent = par
    create_par(root.right, root)
    create_par(root.left, root)
    
    
def DFS(node,res):
    if (node.count == 1):
        res.append(node)
        return
    DFS(node.left,res)
    DFS(node.right,res)


def LCA(node1, node2):
    parent_list = []
    par1 = node1.parent
    while par1 is not None:
        parent_list.append(par1)
        par1 = par1.parent
    par2 = node2.parent
    while par2 not in parent_list:
        par2 = par2.parent
    return par2

def purity(root, cla, target):
    nodes_list = []
    DFS(root, nodes_list)
    target_node = []
    for node in nodes_list:
        if target[node.id] == cla:
            target_node.append(node)
    if len(target_node) == 0:
        return 1
    p = 0
    for i in range(len(target_node)):
        for j in range(i, len(target_node)):
            count = 0
            node1 = target_node[i]
            node2 = target_node[j]
            lca = LCA(node1, node2)
            subtree = []
            DFS(lca, subtree)
            for node in subtree:
                if target[node.id] == cla:
                    count += 1
            p+=(count / len(subtree))
    p /= (len(target_node) * (len(target_node) + 1)) / 2
    return p

def compute_purity(Z, target, target_num = 10):
    rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
    root = create_tree(rootnode)
    create_par(root, None)
    p = 0
    for i in range(target_num):
        p += purity(root, i, target)
    return p/target_num


# ------ random cut related ------

class Binary_Tree():
    
    def __init__(self, left = None, right = None):
        assert isinstance(left, Binary_Tree) or (left == None) or isinstance(left, Leaf_node)
        assert isinstance(right, Binary_Tree) or (right == None) or isinstance(right, Leaf_node)
        self.right = right
        self.left = left
        
    def set_right(self, right):
        assert isinstance(right, Binary_Tree) or (right == None) or isinstance(right, Leaf_node)
        self.right = right
        
    def set_left(self, left):
        assert isinstance(left, Binary_Tree) or (left == None) or isinstance(left, Leaf_node)
        self.left = left
        
class Leaf_node():
    def __init__(self, value):
        self.value = value
        
def random_cut(n, data_list):
    if n == 1:
        return Leaf_node(data_list[0])
    else:
        a1 = min(data_list)
        an = max(data_list)
        data_right = []
        data_left = []
        m = 0
        r = np.random.uniform(a1, an)
        for d in data_list:
            if d < r:
                data_left.append(d)
                m += 1
            else:
                data_right.append(d)
        x = Binary_Tree()
        x.set_right(random_cut(n - m, data_right))
        x.set_left(random_cut(m, data_left))
        return x
    
def list_leaves(node):
    if node.is_leaf():
        return [node.id]
    else:
        result = []
        result += list_leaves(node.left)
        result += list_leaves(node.right)
        return result
    
    
def Gaussian_similarity(x1, x2):
    return np.exp(-1 / 2 * (x1 - x2)**2)

def compute_objective(root, max_obj):
    obj = 0
    if isinstance(root, Leaf_node):
        return 0
    else:
        right_leaves  = list_leaves(root.right)
        left_leaves = list_leaves(root.left)
        for i, xi in enumerate(right_leaves):
            for j in range(i + 1, len(right_leaves)):
                xj = right_leaves[j]
                obj += len(left_leaves) * Gaussian_similarity(xi, xj)
                
        for i, xi in enumerate(left_leaves):
            for j in range(i + 1, len(left_leaves)):
                xj = left_leaves[j]
                obj += len(right_leaves) * Gaussian_similarity(xi, xj)
                
                
        obj_right = compute_objective(root.right, max_obj)
        obj_left = compute_objective(root.left, max_obj)
        #print(obj, obj_right, obj_left)
        return obj + obj_right + obj_left
    
def compute_objective_plus(n, root):
    obj = 0
    if isinstance(root, Leaf_node):
        return 0
    else:
        right_leaves  = list_leaves(root.right)
        left_leaves = list_leaves(root.left)
        for i, xi in enumerate(right_leaves):
            for j, xj in enumerate(left_leaves):
                obj += (n - len(left_leaves) - len(right_leaves)) * Gaussian_similarity(xi, xj)
                
        obj_right = compute_objective_plus(n, root.right)
        obj_left = compute_objective_plus(n, root.left)
        #print(obj, obj_right, obj_left)
        return obj + obj_right + obj_left 
    
def compute_objective_gt(n, root, cla):
    obj = 0
    if root.is_leaf():
        return 0
    else:
        right_leaves  = list_leaves(root.right)
        left_leaves = list_leaves(root.left)
        for i, index1 in enumerate(right_leaves):
            for j, index2 in enumerate(left_leaves):
                xi = cla[index1]
                xj = cla[index2]
                #if xi == xj:
                obj += (n - len(left_leaves) - len(right_leaves)) * (xi == xj)
                
        obj_right = compute_objective_gt(n, root.right, cla)
        obj_left = compute_objective_gt(n, root.left, cla)
        #print(obj, obj_right, obj_left)
        return obj + obj_right + obj_left     

    
def compute_objective_increment(root, whole_data, dic):
    if isinstance(root, Leaf_node):
        return 0
    revenue = 0
    right_leaves  = list_leaves(root.right)
    left_leaves = list_leaves(root.left)
    revenue += (len(left_leaves) + 1) * compute_revenue(right_leaves, whole_data, dic)
    revenue += (len(right_leaves) + 1) * compute_revenue(left_leaves, whole_data, dic)
    rev_right = compute_objective_increment(root.right, whole_data, dic)
    rev_left = compute_objective_increment(root.left, whole_data, dic)
    return revenue + rev_right + rev_left

def max_objective(data):
    data = np.sort(data)
    result = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            for k in range(j + 1, len(data)):
                result += max(Gaussian_similarity(data[i], data[j]), Gaussian_similarity(data[j], data[k]))
                
    return result 

def min_objective(data):
    n = len(data)
    data = np.sort(data)
    result = 0
    for i in range(n - 1):
        summant = 0
        for j in range(i + 1):
            summant += Gaussian_similarity(data[i + 1], data[j]) 
        result += (n - i - 2) * summant
    return result

def sum_objective(data):
    data = np.sort(data)
    result = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            for k in range(j + 1, len(data)):
                result += Gaussian_similarity(data[i], data[j]) + Gaussian_similarity(data[j], data[k])
                
    return result 

def estimate_variance(data):
    result = 0
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            for k in range(j + 1, len(data)):
                result += 1 / 4 * (Gaussian_similarity(data[i], data[j]) - Gaussian_similarity(data[j], data[k])) ** 2
    return result

def run_random_cut(data, max_obj):    
    root = random_cut(len(data), data)
    return compute_objective_plus(len(data), root)

def gen_data(k, n, mu, sigma):
    data = np.array([])
    for i in range(k):
        data = np.concatenate((np.random.normal(mu[i], sigma[i], n), data))
    return data

def preprocess(data):
    data = np.sort(data)
    n = len(data)
    result = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            result[i, j] = Gaussian_similarity(data[i], data[j])
    return result


def compute_revenue(data):
    n = len(data)
    result = 0
    for i in range(n):
        for j in range(i + 1, n):
            result += Gaussian_similarity(data[i], data[j])
    return result

def random_cut_plus(n, data_list, whole_data, dic):
    if n == 1:
        return Leaf_node(data_list[0])
    else:
        a1 = min(data_list)
        an = max(data_list)
        max_revenue = 0
        max_r = 0
        max_data_right = []
        max_data_left = []
        for i in range(16):
            revenue = 0
            r = np.random.uniform(a1, an)
            m = 0
            data_right = []
            data_left = []
            for d in data_list:
                if d <= r:
                    data_left.append(d)
                    m += 1
                else:
                    data_right.append(d)
            revenue += m * compute_revenue(data_right)
            revenue += (n - m) * compute_revenue(data_left)
            #print(m, n, compute_revenue(data_right, whole_data, dic), compute_revenue(data_left, whole_data, dic))
            if revenue > max_revenue:
                max_revenue = revenue
                max_r = r
                max_data_right = data_right
                max_data_left = data_left
        if max_revenue == 0:
            max_data_right = data_right
            max_data_left = data_left

        x = Binary_Tree()
        m = len(max_data_left)
        x.set_right(random_cut_plus(n - m, max_data_right, whole_data, dic))
        x.set_left(random_cut_plus(m, max_data_left, whole_data, dic))
        return x
    
def random_cut_plus_with_objective(n, data_list, whole_data, dic):
    if n == 1:
        return Leaf_node(data_list[0]), 0
    else:
        a1 = min(data_list)
        an = max(data_list)
        revenue = 0
        r = np.random.uniform(a1, an)
        m = 0
        data_right = []
        data_left = []
        for d in data_list:
            if d <= r:
                data_left.append(d)
                m += 1
            else:
                data_right.append(d)
        revenue += m * compute_revenue(data_right)
        revenue += (n - m) * compute_revenue(data_left)
        #print(m, n, compute_revenue(data_right, whole_data, dic), compute_revenue(data_left, whole_data, dic))

        x = Binary_Tree()
        m = len(data_left)
        right_node, rev_right = random_cut_plus_with_objective(n - m, data_right, whole_data, dic)
        left_node, rev_left = random_cut_plus_with_objective(m, data_left, whole_data, dic)
        x.set_right(right_node)
        x.set_left(left_node)
        return x, revenue + rev_right + rev_left
    
def run_random_cut_plus(data, max_obj, dic = None): 
    if dic is None:
        dic = preprocess(data)
    root = random_cut_plus(len(data), data, data, dic)
    return compute_objective_plus(len(data), root)

def run_experiment(n, k):
    data = gen_data(k, n // k, [2 * i for i in range(k)], np.random.uniform(0,1,k))
    dic = preprocess(data)
    max_obj = max_objective(data)
    #sum_obj = sum_objective(data)
    est_var = estimate_variance(data)
    obj1 = []
    obj2 = []
    for i in range(100):
        obj1.append(run_random_cut(data, max_obj)/ max_obj)
        obj2.append(run_random_cut_plus(data, max_obj, dic)/ max_obj)
    obj1 = np.array(obj1)
    obj2 = np.array(obj2)
    return obj1, obj2

def random_cut_1D(n, data_list, index):
    if n == 1:
        return Leaf_node(index[0])
    else:
        a1 = min(data_list)
        an = max(data_list)
        data_right = []
        data_left = []
        index_right = []
        index_left = []
        m = 0
        r = np.random.uniform(a1, an)
        for i, d in enumerate(data_list):
            if d < r:
                data_left.append(d)
                index_left.append(index[i])
                m += 1
            else:
                data_right.append(d)
                index_right.append(index[i])
        x = Binary_Tree()
        x.set_right(random_cut_1D(n - m, data_right, index_right))
        x.set_left(random_cut_1D(m, data_left, index_left))
        return x

def projected_random_cut(data):
    n, d = data.shape
    dir = np.random.randn(d, 1)
    norm = np.linalg.norm(dir)
    dir = dir / norm
    new_data = data.dot(dir)
    index = np.arange(n)
    root = random_cut_1D(n, new_data, index)
    return root