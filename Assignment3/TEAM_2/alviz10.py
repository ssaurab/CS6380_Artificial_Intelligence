import sys
import heapq
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication,QPushButton
from PyQt5.QtGui import QIcon

from PyQt5.QtWidgets import QWidget, QApplication,QLabel, QPlainTextEdit
from PyQt5.QtGui import QPainter,QPen
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QSize    
import sys, random
import pickle
import random
import numpy as np
import scipy.spatial
from PyQt5.QtCore import QPoint
import collections
import time

import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
# screen_width = 1750
screen_height = root.winfo_screenheight()
# screen_height = 950
closed_set=set()
open_set = set()
n_node = 1000
t_coord_in={}
t_in_coord={}
realrelay = -1
no_iterations = -1
first_iter = 1
first_iter_c = 1
first_iter_o = 1
relay_set = set()
foofile = open("dc.txt","w+")
# adjacency_list =[]

# node_colour = [0 for i in range(n_node)] # 0  - not visited, 1 - open list, 2 - closed set, 3- Start node, 4 - Goal node

# def generate_points(xl,yl,number=100):
def generate_points(xl,yl,number=50):
    x_coordinates = np.random.randint(xl, size=number)
    y_coordinates = np.random.randint(yl, size=number)
    t=[]
    for i,j in zip(list(x_coordinates),list(y_coordinates)):
        t.append([i,j+70])

    
    return t

def make_edge_list_tsp(node,n_nodes):
    edg = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if (i<j):
                edg.append([(node[i][0],node[i][1]),(node[j][0],node[j][1])])

    with open('edge_list.pkl', 'wb') as f:
        pickle.dump(edg, f)    

    return edg             


def find_neighbors(pindex, triang):
    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]

# return list of list of tuples Ex.[[(x1,y1),(x2,y2)]]
def make_edge_list(node,n_node,tri,bf):
    temp=[]
    for k in range(n_node):
        pindex = k
        neighbor_indices = find_neighbors(pindex,tri)
        for i in range(len(neighbor_indices)):
            # if i%2!=0:
            if i%bf!=0:
                temp.append([(node[pindex][0],node[pindex][1]),(node[neighbor_indices[i]][0],node[neighbor_indices[i]][1])])    
    with open('edge_list.pkl', 'wb') as f:
        pickle.dump(temp, f)
    return temp

# Make adjacency list 
def make_adj_list(a,d,n):
    t=[[]for i in range(n+1)]
    tt=[]
    for i in a:
        t[d[i[0]]].append(d[i[1]])
        t[d[i[1]]].append(d[i[0]])
    for i in t:
        tt.append(list(set(i)))
    return tt


def make_dict_index_coord(points):
    # t = {}
    t_in_coord={}
    # print("points in in - co")
    # print(points)
    cnt=1
    for i in points:
        t_in_coord[cnt] = (i[0],i[1])
        cnt+=1
    return t_in_coord


# Mapping co-ordinates to node index.
def make_dict_coord_index(points):
    # t={}
    t_coord_in={}
    # print("points in co - in")
    # print(points)
    cnt=1
    for i in points:
        t_coord_in[(i[0],i[1])] = cnt
        cnt+=1
    return t_coord_in
                
# Make dictionary for indexing nodes.
def make_dict_node(points):
    dict_node = {}
    cnt=1
    for i in points:
        dict_node[1] = i
        cnt+=1
    return make_dict_node

def ret_edg():
    with open('edge_list.pkl', 'rb') as f:
            edg = pickle.load(f)
    return edg

# def make_dict_index_coord(points):
#     t = {}
#     cnt=1
#     for i in points:
#         t[cnt] = (i[0],i[1])
#         cnt+=1
#     return t

def movegen(adj,n_ind):
    return adj[n_ind]

def backtrace(parent, start, end):
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

goal_copy = -1
U = 0


def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 


def distance(node1, node2):


    #print(dict_index_coord[node1])
    a1,b1 = dict_index_coord[node1]
    a2,b2 = dict_index_coord[node2]
    # print (np.sqrt((a1-a2)*(a1-a2) + (b1-b2)*(b1-b2)))
    return np.sqrt((a1-a2)*(a1-a2) + (b1-b2)*(b1-b2))

def f(curr_node):
    global goal_copy
    goal_node = goal_copy
    return dijk_distances[curr_node] + distance(curr_node,goal_node)

def heuristic(curr_node):

    global goal_copy
    goal_node = goal_copy

    #print(dict_index_coord[goal_node])
    a1,b1 = dict_index_coord[curr_node]
    a2,b2 = dict_index_coord[goal_node]
    #print (np.sqrt((a1-a2)*(a1-a2) + (b1-b2)*(b1-b2)))
    return np.sqrt((a1-a2)*(a1-a2) + (b1-b2)*(b1-b2))

beam_stack = []
open_layers = [[]]
closed_layers = [[]]
dijk_distances = [0 for i in range(10000)]#but n_node is only 1000, what if it is more? 
parent_memorize_test = []
actual_dijk_answers = [10000 for i in range(10000)]
relay_association = []


def prune(n,w,last_closed):

    global parent_memorize_test
    global beam_stack
    open_layers[n].sort(key = f)
    keep = open_layers[n][:w-1]

    prune = Diff(open_layers[n], keep)
    # print("removing:")
    # print(prune)
    node = min(prune,key=f)
    # print("g value of the least f node is : ")
    # print(dijk_distances[node])
    a, b = beam_stack[n-1]
    beam_stack[n-1] = a, f(node)
    # print(beam_stack[n-1])


    for pn in prune:
        open_layers[n].remove(pn)


    for a,b in parent_memorize_test:
        for pn in prune:
            if a == pn and b in last_closed:
                parent_memorize_test.remove((a,b))


def solutionreconstruct(start_node, goal_node):
    path = []
    path = [start_node, adjacency_list[start_node][0], adjacency_list[adjacency_list[start_node][0]][0]]
    return path

def dumbbfs(start_node, goal_node):


    visited = []
    Q = [start_node]
    parents=[]
    while Q: 
        vertex = Q.pop(0)
        visited.append(vertex)
        for neighbour in adjacency_list[vertex]: 
            if neighbour == goal_node:
                parents.append((neighbour,vertex))
                Q=[]
                break
            if neighbour not in visited: 
                visited.append(neighbour) 
                parents.append((neighbour, vertex))
                Q.append(neighbour)

    # print("right heere")[1, 7]
    need_parent = neighbour
    path = []
    while need_parent != start_node:
        path.append(need_parent)
        if len(path) > 8:
            return path
        # print(path)
        for a,b in parents:
            if a == need_parent:
                need_parent = b 

    # print("not yet here")
    path.append(start_node)
    path.reverse()
    return path



def calculate_cost(path):
    sum = 0
    prev = path[0]
    for n in path:
        sum += distance(prev,n)
        prev = n

    return sum




def beamstacksearch(start_node, goal_node, w, relay):
    global U
    global open_layers
    global closed_layers
    global parent_memorize_test
    global beam_stack
    global open_set
    global closed_set
    global relay_association
    global relay_set
    global node_colour
    open_layers = [[] for i in range(10000)]
    closed_layers = [[] for i in range(10000)]
    parent_memorize_test = []
    open_layers[0].append(start_node)
    global dijk_distances
    # closed_set.clear()
    dijk_distances = [0 for i in range(10000)]
    best_goal = -1
    gotagoal = -1

    #should the best_goal also be global?
    # print(start_node)
    l = 0
    x = 0
    # print("testing the implementation of argmin")
    # print(min(open_layers[l],key=heuristic))
    while open_layers[l] or open_layers[l+1]:
        while open_layers[l]:
            x += 1
            # print("Im inside the double while now")
            node = min(open_layers[l],key=f)
            # print("the best node currently in open is : ")
            # print(node)
            open_layers[l].remove(node)
            closed_layers[l].append(node)
            # closed_set.add(node)
            # for n in closed_set:
            #   node_colour[n] = 2

            if node == goal_copy:
                #U = heuristic(node)
                #update this after writing solution reconstruction method
               
                gotagoal = goal_copy
                for n in closed_layers[l-1]:
                    if first_iter_c == 1:
                        closed_set.add(n)
                for n in closed_set:
                    node_colour[n] = 2


                # open_set = set()
                for n in open_layers[l]:
                    if first_iter_o == 1:
                        open_set.add(n)
                        node_colour[n] = 4

                # return [start_node] #make sure to change it later
            for n in adjacency_list[node]:
                # a,b = beam_stack[len(beam_stack)-1] : a very important diversion from pseudocode!! may need to look back
                a,b = beam_stack[l]
                if dijk_distances[node]+distance(n, node) < dijk_distances[n] or dijk_distances[n] == 0:
                    expected_distance = dijk_distances[node]+distance(n, node)
                else:
                    expected_distance = dijk_distances[node]
                f_n = expected_distance + distance(n, goal_copy)
                # print("expected fn is : "),
                # print(f_n, a, b)
                # print("gn will be : "),
                # print(expected_distance)
                if f_n < b and f_n > a and n not in closed_layers[l] and n not in closed_layers[l-1] and n not in closed_layers[l+1] and n not in open_layers[l] and n not in open_layers[l-1] and n not in open_layers[l+1] :
                    
                    # if (node, n) not in parent_memorize_test:
                    
                    if not dijk_distances[n] or dijk_distances[n] > dijk_distances[node] + distance(node, n):

                        for a,b in parent_memorize_test:
                            if a == n:
                                parent_memorize_test.remove((a,b))
                        parent_memorize_test.append((n, node))
                        if l == relay:
                            for a,b in relay_association:
                                if a == n:
                                    relay_association.remove((a,b))
                            
                            relay_association.append((n, node))

                            if first_iter == 1:
                                relay_set.add(node)
                            # for n in relay_set:
                                node_colour[node] = 3

                        if l > relay:
                            relay_ancestor = -1
                            for a,b in relay_association:
                                if a == node:
                                    relay_ancestor = b
                            for a,b in relay_association:
                                if a == n:
                                    relay_association.remove((a,b))
                            relay_association.append((n,b))
                            
                            if first_iter == 1:
                                relay_set.add(b)
                        
                        dijk_distances[n] = dijk_distances[node] + distance(node, n)
                        open_layers[l+1].append(n)
                        # open_set.add(n)
                        # for node_open_show in open_set:
                        #   node_colour[node_open_show] = 1

                        # print("adding pointer :")
                        # print((n,node))
                    # else:
                    #      print("skipping"),
                    #      print((n,node))
                    # print("adding")
                    # print(f_n)
                    # print(n)
                    # print(adjacency_list[n])

                if f_n < b and f_n > a and n in open_layers[l+1]:
                    #this should handle in beam parent ambiguity
                    if dijk_distances[n] > dijk_distances[node] + distance(node, n):
                        dijk_distances[n] = dijk_distances[node] + distance(node, n)
                        for a,b in parent_memorize_test:
                            if a == n and b in closed_layers[l]:
                                parent_memorize_test.remove((n, b))
                                relay_association[n] = relay_association[b]
                        parent_memorize_test.append((n, node))
                if f_n < b and f_n > a:
                    #this should handle in beam parent ambiguity
                    if dijk_distances[n] > dijk_distances[node] + distance(node, n):
                        dijk_distances[n] = dijk_distances[node] + distance(node, n)
                        for a,b in parent_memorize_test:
                            if a == n:
                                parent_memorize_test.remove((n, b))
                                relay_association[n] = relay_association[b]
                        parent_memorize_test.append((n, node))
                # if heuristic(n) < b and heuristic(n) >=a and n in open_layers[l+1]:
                #     #to handle parent change of existing nodes in the open layer
                #     path_to_return = []
                #     need_parent = node
                #     actual_parent = -1

                #     while need_parent != start_node:
                #         path_to_return.append(need_parent)

                #         for a,b in parent_memorize_test:
                #             if a == need_parent:
                #                 need_parent = b 
                #             if a == n:
                #                 actual_parent = b
                #     print("not yet here")
                #     path_to_return.append(start_node)
                #     path_to_return.reverse()

                #     path_to_return2 = []
                #     need_parent = actual_parent

                #     while need_parent != start_node:
                #         path_to_return2.append(need_parent)

                #         for a,b in parent_memorize_test:
                #             if a == need_parent:
                #                 need_parent = b 
                   
                #     print("not yet here")
                #     path_to_return2.append(start_node)
                #     path_to_return2.reverse()



                #     tt1 = calculate_cost(path_to_return) + distance(node,n)
                #     tt2 = calculate_cost(path_to_return2)+ distance(actual_parent, n)

                #     if tt1 <= tt2:
                #         parent_memorize_test.remove(n, actual_parent)
                #         parent_memorize_test.append(n, node)

                # if f_n >= b :
                #     print(n),
                #     print("overshot the limit")
                # if f_n <a:
                #     print(n),
                #     print("has heuristic lower than needed")
            if len(open_layers[l+1]) > w-1:
                prune(l+1,w,closed_layers[l])
            # print(beam_stack)


        if l > 1 and l <= relay:
            closed_layers[l-1] = []
        if l > relay+1:
            closed_layers[l-1] = []
        # If these were present, there was a possibility of a loop in the answer!!

        l += 1
        open_layers[l+1] = []
        closed_layers[l] = []
        lastelem = beam_stack[len(beam_stack)-1]
        if(len(beam_stack) <= l):
            beam_stack.append((0,U))
        # print(beam_stack[len(beam_stack)-1])






    if gotagoal != -1:
        # return solutionreconstruct(start_node, goal_node)
        # print("found best goal")
        best_goal = node

        # for a, b in relay_association:
        #     if a == goal_copy:
        #         node_colour[b] = 3
        # # node_colour[relay_association[goal_copy]] = 3
        path_to_return = []
        need_parent = goal_copy
        ttt = 0
        while need_parent != start_node and ttt < 50:
            ttt += 1
            path_to_return.append(need_parent)
            # if len(path_to_return) > 8:
                # return path_to_return
            # print(path)
            for a,b in parent_memorize_test:
                if a == need_parent: 
                    need_parent = b 
                    # print(need_parent)
                    # print(parent_memorize_test)

        # print("not yet here")
        # print(dijk_distances)
        path_to_return.append(start_node)
        path_to_return.reverse()
        return path_to_return, dijk_distances[goal_node]
    else :
        return [],U


def findmin(lista):

    a = -1
    b = 100000
    for p, q in lista:
        if b > q:
            b = q
            a = p
    return a,b

def sortdoublelist(x , y):
    a, b = x
    c, d = y
    return a < c
printed = -1
def actual_dijk(starting_vertex, n_node):
    # print("number of nodes is :")
    # print(n_node)
    global printed
    distances = {vertex: float('infinity') for vertex in range (1,n_node+1)}
    distances[starting_vertex] = 0

    entry_lookup = {}
    pq = [(starting_vertex,0)]
    fixvertices = []
    fixvertices.append(starting_vertex)
    workingmemory = []
    
    while len(pq) != n_node:
        for a,b in pq:
            for c in adjacency_list[a]: 
                if c not in fixvertices:
                    workingmemory.append((c, b+distance(c,a)))
        # ll = [[]]
        # ll.append(workingmemory)
        a,b = findmin(workingmemory)

        pq.append((a,b))
        fixvertices.append(a)
        workingmemory = []

    pq.sort(key=lambda x: x[0])
    # print (pq)
    if printed != 1:
        foofile.write(str(pq))
    printed = 1



def Algorithm(adjacency_list,start_node,goal_node,n_node):
#you have to add the assigned algorithm here
    # print(n_node)
    # print("is the number of nodes present")
    global closed_set
    global open_set
    global dict_coord_index
    global goal_copy
    global relay_association
    global U
    global beam_stack

    w = 5
    goal_copy = goal_node
    relay = 3
    parent_memorize_test = []
##################################

    #This is where I write my algorithm
    #inferences : n_node is the number of nodes 100,500 etc
    #Doubts : what if multiple nodes have the same heuristic value?

##################################

    relay_association = [(i,-1) for i in range(10000)]
    index_dumb = 0
    for i in adjacency_list:
        # print (index_dumb),
        # print (i),
        index_dumb += 1
    # print(start_node)
    # print("is the start")
    # print(goal_node)
    # print("is the goal node")

    # print("the children of start node are :")
    # print(adjacency_list)


    # print(dict_coord_index)
    # actual_dijk(start_node, n_node)
    heuristic(start_node)

    # raw_input()
    #list can be used as a stack
    #only use the methods : stack pop, stack append

    U = 2500 #need to decide based on the max pixels

    beam_stack = []
    beam_stack.append((0,U))
    # print(beam_stack)

    dijk_distances[start_node] = 0
    #working : print(beam_stack)
    path_answer=[]# this is the path or tour to be returned
    last_path_val = 100000

    while beam_stack:

        # print(beam_stack)
        path, newU = beamstacksearch(start_node, goal_node,w, relay)
        if path:
            # path = dumbbfs(start_node, goal_node)
            # print("the method returned a path, which is : ##############################")
            # print(path)
            if not path_answer:
                path_answer = path
                last_path_val = newU
                U = newU
            if path_answer:
                a1= last_path_val
                a2= newU
                # print("showing path costs : ")
                # print(a1, a2)
                if a1 >= a2:
                    path_answer = path
                    U = a2

        # print(beam_stack[len(beam_stack)-1])
        # print(U)

        while beam_stack and beam_stack[len(beam_stack)-1][1] >= U :
            # print("pops are happening!")
            # print(beam_stack[len(beam_stack)-1])
            beam_stack.pop()

        if not beam_stack:
            return path_answer

        a,b = beam_stack[len(beam_stack)-1]
        # print("change in algo")
        # print("U is : "),
        # print(U)
        # print(parent_memorize_test)
        # print(beam_stack[len(beam_stack)-1])
        beam_stack[len(beam_stack)-1] = (b,U)
        if b==U:
             break
            #edit this part later

    # print("done!")
    return path_answer


def my_main(n_node = 100, bf = 2,gg=0):
    x_dim = screen_width-100
    y_dim = screen_height-150
    # edge_list=pickle.load(f)
    with open('edge_list.pkl', 'rb') as f:
            edge_list = pickle.load(f)
    
    global dict_index_coord
    global dict_coord_index
    global adjacency_list 

    if (gg==0):
        node = generate_points(x_dim,y_dim,n_node)
        # global dict_index_coord
        dict_index_coord = make_dict_index_coord(node) # indexing to pixels pair. Ex. 1->(10,10)
        # global dict_coord_index
        dict_coord_index = make_dict_coord_index(node) # mapping pixels to index pair. Ex. (10,10) -> 1
        # print("MAP")
        # print(dict_coord_index)
        # print(dict_index_coord)
        tri = scipy.spatial.Delaunay(np.array(node))
        
        edge_list = make_edge_list(node,n_node,tri,bf)            
        # print(edge_list)
        # global adjacency_list 
        adjacency_list= make_adj_list(edge_list,dict_coord_index,n_node) #adjcency list             
        
        return edge_list
    elif (gg==2):
        node = generate_points(x_dim,y_dim,n_node)
        # global dict_index_coord
        dict_index_coord = make_dict_index_coord(node) # indexing to pixels pair. Ex. 1->(10,10)
        # global dict_coord_index
        dict_coord_index = make_dict_coord_index(node) # mapping pixels to index pair. Ex. (10,10) -> 1
        # print("MAP")
        # print(dict_coord_index)
        # print(dict_index_coord)
        edge_list = make_edge_list_tsp(node,n_node)
        # global adjacency_list 
        adjacency_list= make_adj_list(edge_list,dict_coord_index,n_node)

        return edge_list

    elif (gg==1):
        # tree code  TODO
        return edge_list


parent_recursive_collector = []

def motherfunction(adjacency_list,start_node,goal_node,n_node):
    global printed
    global parent_recursive_collector
    global foofile
    global no_iterations
    global first_iter
    global first_iter_c
    global first_iter_o
    global node_colour
    global open_set
    global closed_set
    global relay_set
    printed = -1
    foofile = open("dc.txt","w+")
    open_set = set()
    closed_set = set()
    relay_set = set()
    pathfirst = Algorithm(adjacency_list, start_node, goal_node, n_node)
    first_iter = -1
    first_iter_c = -1
    first_iter_o = -1
    realrelay = len(pathfirst)/2
    divideandconquer(adjacency_list,start_node,goal_node,n_node, realrelay)
    pathcache = []
    no_iterations = len(pathfirst)
    parent_recursive_collector.reverse()
    # print(parent_recursive_collector)
    # print(pathfirst)
    need_parent = goal_node
    # print("needs parent : "),
    # print(need_parent)
    while need_parent != start_node:
        for a,b in parent_recursive_collector:
            if a == need_parent and a != start_node:
                need_parent = b
                # print("appending : "),
                # print(a)
                pathcache.append(a)
            if a == start_node:
                break
    for n in relay_set:
        node_colour[n] = 3
    for n in closed_set:
        node_colour[n] = 2
    for n in open_set:
        node_colour[n] = 1
    pathcache.append(start_node)
    pathcache.reverse()
    # print("relay : ")
    # print(relay_set)
    # print(pathcache)
    # print(calculate_cost(pathcache))
    return pathcache


def divideandconquer(adjacency_list,start_node,goal_node,n_node, realrelay):
    global parent_recursive_collector
    foofile.flush()
    if goal_node in adjacency_list[start_node]:
        # print("******************************************************")
        # print("adding nodes into parent parent_recursive_collector : ")
        # print(goal_node),
        # print(start_node)
        parent_recursive_collector.append((goal_node, start_node))
        # foofile.write('\n')
        # foofile.write("basecase : ")
        # foofile.write(str(start_node))
        # foofile.write(" ")
        # foofile.write(str(goal_node))
        # foofile.write(" ")
        # foofile.write(str(realrelay))
        return
    elif start_node == goal_node:
        # foofile.write('\n')
        # foofile.write("basecase2 : ")
        # foofile.write(str(start_node))
        # foofile.write(" ")
        # foofile.write(str(goal_node))
        # foofile.write(" ")
        # foofile.write(str(realrelay))
        return
    else:
        pathnext = Algorithm(adjacency_list,start_node,goal_node,n_node)
        relaynode = pathnext[int(realrelay)]
        # foofile.write('\n')
        # foofile.write("showing path, cost, start, goal : ")
        # foofile.write(str(pathnext))
        # foofile.write(" ")
        # foofile.write(str(calculate_cost(pathnext)))
        # foofile.write(" ")
        # foofile.write(str(start_node))
        # foofile.write(" ")
        # foofile.write(str(goal_node))
        # foofile.write(" ")
        # foofile.write(str(realrelay))

        divideandconquer(adjacency_list,start_node,relaynode,n_node, max(int(realrelay/2), 1))
        divideandconquer(adjacency_list,relaynode,goal_node,n_node, max(int(realrelay/2), 1))
        return


class Example(QMainWindow):
    
    def __init__(self):
        super().__init__()


        self.nodes = 100
        self.bf = 2
        self.dict_index_coord = {}
        self.open_list = []
        self.closed_list = []    
        self.init_phase = -1
        self.start_x=0
        self.start_y=0
        
        self.goal_x=0
        self.goal_y=0
        
        self.initUI()
        self.setMinimumSize(QSize(screen_width,screen_height))    
        self.setWindowTitle("Alviz v0.2") 
       


        
        
    def initUI(self):               
        
        self.exitAct = QAction( '&Exit', self)        
        self.exitAct.setShortcut('Ctrl+Q')
        self.exitAct.setStatusTip('Exit application')
        self.exitAct.triggered.connect(qApp.quit)

        self.genAct = QAction( '&Generate Graph', self)
        self.genAct.triggered.connect(self.clickMethod)
        self.genTreeAct = QAction( '&Generate Tree', self)
        self.genTreeAct.triggered.connect(self.clickMethod1)
        self.genTSPAct = QAction( '&Generate TSP', self)
        self.genTSPAct.triggered.connect(self.clickMethod2)
        self.startAct = QAction( '&Start Node', self)
        self.goalAct = QAction( '&Goal Node', self)
        self.genRevertAct = QAction( '&Revert', self)
        self.genRevertAct.triggered.connect(self.clickMethodRevert)


        self.nodeLabel = QLabel('Number of nodes:')
        self.nodeText = QPlainTextEdit('100')
        self.nodeText.setFixedSize(80,28)
        self.bfLabel = QLabel('Branching Factor:')
        self.bfText = QPlainTextEdit('2')
        self.bfText.setFixedSize(80,28)

        self.resetAct =  QAction( '&Reset Screen', self)
        self.resetAct.triggered.connect(self.reset_screen)


        self.menubar = self.menuBar()
        self.fileMenu = self.menubar.addMenu('File')
        self.fileMenu.addAction(self.exitAct)
        self.toolbar = self.addToolBar('')

        self.toolbar.addWidget(self.nodeLabel)
        self.toolbar.addWidget(self.nodeText)
        self.toolbar.addWidget(self.bfLabel)
        self.toolbar.addWidget(self.bfText)

        self.toolbar.addAction(self.genAct)
        self.toolbar.addAction(self.genTreeAct)
        self.toolbar.addAction(self.genTSPAct)
        self.toolbar.addAction(self.startAct)
        self.toolbar.addAction(self.goalAct)
        self.toolbar.addAction(self.genRevertAct)
        self.toolbar.addAction(self.resetAct)
        
        self.setMouseTracking(True)
        self.startAct.setEnabled(False)
        self.goalAct.setEnabled(False)
        self.path_t=[]

        # my_main()

    def reset_screen(self):
        self.init_phase = -1
        self.startAct.setEnabled(False)
        self.goalAct.setEnabled(False)
        self.update()
        

    def clickMethod(self):
        self.init_phase = 0
        # print('Clicked Pyqt button.')
        self.nodes = int(self.nodeText.toPlainText())
        self.bf = int(self.bfText.toPlainText())
        global node_colour
        node_colour = [0 for i in range(self.nodes+1)] 
        my_main(self.nodes,self.bf,0)
        self.update()

    def clickMethod1(self):
        self.init_phase = 0
        # print('Clicked Pyqt button. 1')
        self.nodes = int(self.nodeText.toPlainText())
        self.bf = int(self.bfText.toPlainText())
        global node_colour
        node_colour = [0 for i in range(self.nodes+1)] 
        my_main(self.nodes,self.bf,1)
        self.update()
        
    def clickMethod2(self):
        self.init_phase = 6
        # print('Clicked Pyqt button. 2')
        self.nodes = int(self.nodeText.toPlainText())
        self.bf = int(self.bfText.toPlainText())
        global node_colour
        node_colour = [0 for i in range(self.nodes+1)] 
        my_main(self.nodes,self.bf,2)
        self.update()
    def clickMethodRevert(self):
        self.init_phase = 0
        print('Clicked Pyqt button. Revert')
        self.nodes = int(self.nodeText.toPlainText())
        self.bf = int(self.bfText.toPlainText())
        global node_colour
        node_colour = [0 for i in range(self.nodes+1)] 
        # my_main(self.nodes,self.bf,2)
        self.update()            
        

    # def mouseClickEvent(self,e):
    def mousePressEvent(self, e):
        x=e.x()
        y=e.y()
        text = "x: {0},  y: {1}".format(x, y)
        min_x= 99999
        max_x=-99999
        min_y= 99999
        max_y=-99999
        self.findClosestCoordinate(min_x,min_y,x,y)
        # self.label.setText(text)
        # print(text)
    
    def findClosestCoordinate(self,min_x,min_y,x,y):
        global dict_coord_index
        edg=ret_edg()
        myset = set()
        min_dist=999999999
        for e in edg: 
            # myset.add(e[0])
            # myset.add(e[1])
            dist=(x-e[0][0])**2+(y-e[0][1])**2
            if(dist<min_dist) :
                min_dist=dist
                min_x=e[0][0]
                min_y=e[0][1]
        print("minimum x :"+str(min_x))
        print("minimum y :"+str(min_y))
        if(self.init_phase==0):#initial phase for planar graph generation
            self.start_x=min_x
            self.start_y=min_y
            self.init_phase=1
            start_node = dict_coord_index[(self.start_x,self.start_y)]#1
            # print(start_node) ##### ALSO REMOVE THESE ASAP!!
            # self.update()
        elif(self.init_phase==1):#After selecting the start node    
            self.goal_x=min_x
            self.goal_y=min_y
            self.init_phase=5
            goal_node = dict_coord_index[(self.goal_x,self.goal_y)]#1
            # print(goal_node)
            # self.update()
        elif(self.init_phase==5):#After selecting the goal node
            # print("start_x , start_y"+str(self.start_x)+","+str(self.start_y))
            start_node = dict_coord_index[(self.start_x,self.start_y)]#1
            print(start_node)
            l=dict_index_coord[start_node]
            # print(l)

            # print("coord - to - int ")
            # print(dict_coord_index)
            # print("int - to - coord ")
            # print(dict_index_coord)
            # print("###################################")
            goal_node =  dict_coord_index[(self.goal_x,self.goal_y)]
            # print(goal_node) ### MAKE SURE TO REMOVE THIS AND START NODE LATER
            self.path_t = motherfunction(adjacency_list,start_node,goal_node,self.nodes)
            # print(node_colour)
            # print("###################################")
            # print (len(node_colour))
            self.init_phase=3
        elif(self.init_phase==3):#show the bfs
            self.init_phase=4            #init_phase=4 is the default end phase of all types of graph
        elif(self.init_phase==6):#initial phase for tsp
            self.init_phase=7
        elif(self.init_phase==7):#final phase for tsp
            self.init_phase=4                        



    def paintEvent(self, e):

        qp = QPainter()
        qp.begin(self)
        # self.print_s(qp)
        if (self.init_phase!=-1):
            self.drawPoints(qp)
            if(self.init_phase!=6):
                self.drawLines(qp)
            qp.end()


    def drawPoints(self, qp):

        qp.setPen(Qt.red)
        size = self.size()


        edg=ret_edg()
        myset = set()
        for e in edg: 
            myset.add(e[0])
            myset.add(e[1])

        xx = list(myset)    
        self.dict_index_coord = make_dict_index_coord(xx)

        # print(self.init_phase)
        # print(self.start_x)
        # print(self.start_y)



        if (self.init_phase == 0):#draw points to create planar graph

            for e in edg :
                center = QPoint(e[0][0],e[0][1])
                qp.setBrush(Qt.yellow)
                qp.drawEllipse(center,5,5)
               #qp.drawPoint(e[0][0],e[0][1])
            
            self.startAct.setEnabled(True)
            
            self.update()

        # else:
        elif (self.init_phase == 1):#draw start node
            for e in edg :
                center = QPoint(e[0][0],e[0][1])
                qp.setBrush(Qt.yellow)
                qp.drawEllipse(center,5,5)

            center = QPoint(self.start_x,self.start_y)
            qp.setBrush(Qt.green)
            qp.drawEllipse(center,10,10)    
            

            self.startAct.setEnabled(False)
            self.goalAct.setEnabled(True)

            self.update()
        elif (self.init_phase == 5):#draw goal node
            for e in edg :
                center = QPoint(e[0][0],e[0][1])
                qp.setBrush(Qt.yellow)
                qp.drawEllipse(center,5,5)

            center = QPoint(self.start_x,self.start_y)
            qp.setBrush(Qt.green)
            qp.drawEllipse(center,10,10)
                
            center = QPoint(self.goal_x,self.goal_y)
            qp.setBrush(Qt.red)
            qp.drawEllipse(center,10,10)    

            self.startAct.setEnabled(False)
            self.goalAct.setEnabled(False)
            self.update()
            # self.init_phase = 5
        elif (self.init_phase == 3):#draw the path and color different nodes as per the color coding mentioned
            i=1
            # print("node col size "+str(len(node_colour)))
            for i in range(1,len(node_colour)):
                point=dict_index_coord[i]
                e = node_colour[i]
                center = QPoint(point[0],point[1])
                if(e==0) :
                    # qp.setBrush(Qt.gray)
                    qp.setBrush(Qt.yellow)
                    qp.drawEllipse(center,5,5)
                if(e==1) :
                    qp.setBrush(Qt.magenta)
                    qp.drawEllipse(center,8,8)
                if(e==2) :
                    qp.setBrush(Qt.blue)
                    qp.drawEllipse(center,8,8)
                if(e==3) :
                    qp.setBrush(Qt.cyan)
                    qp.drawEllipse(center,7,7)
                if(e==4) :
                    qp.setBrush(Qt.red)
                    qp.drawEllipse(center,10,10)
                # i=i+1                    
                # qp.drawEllipse(center,5,5)


                
                self.update()
    
        elif(self.init_phase == 4):    #default final state of all graph
            self.update()
        elif(self.init_phase == 6): #initial phase of tsp
            for e in edg :
                center = QPoint(e[0][0],e[0][1])
                qp.setBrush(Qt.yellow)
                qp.drawEllipse(center,5,5)
               #qp.drawPoint(e[0][0],e[0][1])
            
            self.startAct.setEnabled(True)
            
            self.update()
        elif(self.init_phase == 7):    #finall tour plot
            for e in edg :#you have to use the tour returned by TSP in place of edg
                center = QPoint(e[0][0],e[0][1])
                qp.setBrush(Qt.red)
                qp.drawEllipse(center,5,5)
               #qp.drawPoint(e[0][0],e[0][1])
            
            self.startAct.setEnabled(True)
            
            self.update()


    def drawLines(self, qp):

        pen = QPen(Qt.black, 2, Qt.SolidLine)

        qp.setPen(Qt.gray)

        with open('edge_list.pkl', 'rb') as f:
            edg = pickle.load(f)
        
           #main()
        for e in edg :
           qp.drawLine(e[0][0],e[0][1],e[1][0],e[1][1])    

        if(self.init_phase == 3):
            pen = QPen(Qt.black, 5, Qt.DashDotLine)
            # qp.setPen(Qt.red)
            # qp.setWidth(10)
            pen.setBrush(Qt.red)
            pen.setWidth(5)
            qp.setPen(pen)
            for i in range(len(self.path_t)-1):
                a=self.path_t[i]
                # print("---- point a --- ")
                # print(a)
                a_pos=dict_index_coord[a]
                b=self.path_t[i+1]
                # print("---- point b --- ")
                # print(b)
                b_pos=dict_index_coord[b]    
                qp.drawLine(a_pos[0],a_pos[1],b_pos[0],b_pos[1])
        
if __name__ == '__main__':


    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())


