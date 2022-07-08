import itertools
import random
import numpy as np

def randomGraph(tau,N,proba):
    T = tau
    V = [i for i in range(N)]
    E = []
    for t in range(T):
        for i1 in range(N):
            for i2 in range(i1+1,N):
                if random.random() < proba:
                    E.append((t,{i1,i2}))
    L = T,V,E
    return L

def sort_time_local_degree(Delta,L,Q): #Total order that prioritize local degree
    (T,V,E) = L
    d = np.zeros((T,len(V)))
    for e in E:
        for v in e[1]:
            d[e[0]][v] += 1
    
    D = []
    for g_e in Q:
        localdegree = 0
        for v in g_e[1]:
            localdegree += d[g_e[0],v]
            D.append((g_e[0],localdegree))
            
    return [x for _,x in sorted(zip(D,Q))]


def sort_time_global_degree(Delta,L,Q): #Total order that prioritize global degree
    (T,V,E) = L
    D = []
    Q_size = len(Q)
    for i1 in range(Q_size):
        globaldegree = 0
        t1,e1 = Q[i1]
        for i2 in range(Q_size):
            t2,e2 = Q[i2]
            if (e1 & e2) != set():
                if abs(t1-t2) < Delta:
                    globaldegree += 1
        D.append((t1,globaldegree))
    
    return [x for _,x in sorted(zip(D,Q))]


def Greedy(Delta,L,sort = False):
    T,V,E = L
    Q = E
    if not sort:
        Q = sorted(Q)
    else:
        Q = sort(Delta,L,Q)
    
    (T,V,E) = L
    M = []
    rho = [[0 for v in range(len(V))]for t in range(T)]
    
    while Q != []:
        current_edge = Q.pop(0)
        (t,e) = current_edge
        
        cond = True
        
        for t_prime in range(t,t+Delta):
            if t_prime < T:
                if not cond:
                    break
                for v in e:
                    if rho[t_prime][v]:
                        cond = False
                        break
        
        if cond:
            M.append(current_edge)
            for t_prime in range(t,t+Delta):
                if t_prime < T:
                    for v in e:
                        rho[t_prime][v] = 1
    return M


def timesort(L,S1,S2):
    T,V,E = L
    SS1 = sorted(S1,reverse = True)
    SS2 = sorted(S2,reverse = True)
    for s1,s2 in zip(SS1,SS2):
        if s1[0] != s2[0]:
            return s1[0] > s2[0]
    return S1 > S2



def order(S1,S2,sort = False,Graph = 0):
    """
    S1 \subset M
    S2 \subset E\M
    return true if S2 ≻ S1 with ≻ total order of 2^E
    """
    
    if len(S2) != len(S1): return len(S2) > len(S1)
    
    maxt2 = max(S2)[0]
    maxt1 = max(S1)[0]
    if maxt1 != maxt2: return maxt1 > maxt2
    
    if (not sort) or (Graph == 0):
        return S2 < S1
    return sort(Graph,S1,S2)

def indep(Delta,e1,e2):
    """
    checks if 2 edges are colliding or not
    """
    if e1[1] & e2[1] == set(): return True
    return abs(e1[0]-e2[0]) >= Delta 

def Yfinder(Delta,X,M):
    """
    X \in E\M
    return Y = {m \in M | \exists x in X, \neg m \indep x}
    """
    Y = []
    for x in X:
        for m in M:
            if not(indep(Delta,x,m)):
                if m not in Y:
                    Y.append(m)
    return Y

def close(Delta,X,t):
    maxt = X[-1][0]
    mint = X[0][0]
    return (maxt-mint) < (t*Delta)

def indeplist(Delta,M): #checks if M is a matching
    length = len(M)
    for i in range(length):
        for j in range(i+1,length):
            if not(indep(Delta,M[i],M[j])):return False
    return True

def pLS(Delta,L,sort = False,p=2,start = False,verbose = False):
    T,V,E = L
    E = sorted(E)
    if not start: 
        M = []
        Q = E
    else: #initialize M to some matching
        M = Greedy(Delta,L,sort = sort_time_global_degree)
        Q = [e for e in E if e not in M]
        
    #Q is a sorted list of E\M
    
    cond = True #do while
    while cond: #stops if there are no swaps 
        cond = False
        breakcond = False 
        for t in range(1,p+1): #size of swaps
            if breakcond: break 
                
            for X in itertools.combinations(Q,t): #enumerates all possible improving set of size t
                X = list(X)
                if close(Delta,X,t): #quick check to see if all edges in X are chronologically close (if not then no need to check)
                    if indeplist(Delta,X): # checks X is a matching
                        Y = Yfinder(Delta,X,M) #list all edges of M that intersect X

                        if order(Y,X,sort,Graph = L): #checks if X is an improving set
                            M = [m for m in M if m not in Y] + X #M <- M\Y U X
                            Q = [e for e in E if e not in M] # update Q
                            breakcond = True
                            cond = True
                            if verbose:
                                print(X)
                                print(Y)
                                print()
                            break
    return M