from math import sqrt

def stat(lst):
    """Calculate mean and std deviation from the input list."""
    n = float(len(lst))
    mean = sum(lst) / n
    stdev = sqrt((sum(x*x for x in lst) / n) - (mean * mean)) 
    return mean, stdev

def parse(lst, n):
    cluster = []
    for i in lst:
        if len(cluster) <= 1:    # the first two values are going directly in
            cluster.append(i)
            continue

        mean,stdev = stat(cluster)
        if abs(mean - i) > n * stdev:    # check the "distance"
            yield cluster
            cluster[:] = []    # reset cluster to the empty list

        cluster.append(i)
    yield cluster           # yield the last cluster

# array = [1, 2, 3, 60, 70, 80, 100, 220, 230, 250]
# array =[1,2,4,7,9,5,4,7,9,56,57,54,60,200,297,275,243,1000]
array = [130, 167, 213, 441, 445, 451, 478, 515, 526, 564, 655, 782, 1261]
for cluster in parse(array, 7):
    print(cluster)