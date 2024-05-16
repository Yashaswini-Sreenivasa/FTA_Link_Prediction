
import numpy as np
import random
import networkx as nx

class Node:
    '''Node object:

    id = int() - unique ID for the node, needs to be unique for the entire dataset 
    node_type = int() ∈ {0, 1, 2} 0 for top event, 1 for intermediate event, 2 for basic event
    depth = int() the depth of the node relative to the top event, top event is depth = 0
    gate_type = int() ∈ {0, 1}, 0 = And-gate, 1 = Or-gate

    features = dictonary of all node features - you can add more to this dictionary 
    children = list of all the child nodes
    parents = list of all the parent nodes

    # For basic events, node_type == 2:
    # failure_probability = float() - failure probability   
    
    '''
    def __init__(self, id, node_type, depth, gate_type):
        self.id = id
        self.node_type = node_type
        self.depth = depth
        self.gate_type = gate_type
        self.children = []
        self.parents = []

        self.features = {'id' : self.id, 
                         'node_type' : self.node_type,
                         'depth' : self.depth,
                         'gate_type' : self.gate_type}                 

class FaultTree:
    '''
    Just a minimal FaultTree Object incase you need to do fault tree level operations (I suspect you might need too)

    If you want you can change this implimentation to adhere to the G = (V, E) scheme.
    '''
    def __init__(self, id, min_children, max_children, max_depth):
        self.id = id
        self.min_children = min_children
        self.max_children = max_children
        self.max_depth = max_depth
        self.nodes = []

        top_event = self.create_top_event()
        self.generate_children(top_event)

    # Functions

    def create_top_event(self):
        gate_type = random.choice([0, 1])
        top_event = Node(self.id, 0, 0, gate_type)
        self.nodes.append(top_event)
        self.id += 1
        return top_event

    def generate_children(self, node):
        if node.depth >= self.max_depth:
            return
        num_children = random.randint(self.min_children, self.max_children)
        for _ in range(num_children):
            gate_type = random.choice([0, 1])
            node_type = 1 if node.depth + 1 < self.max_depth else 2
            child_node = Node(self.id, node_type, node.depth + 1, gate_type)
            node.children.append(child_node)
            child_node.parents.append(node)
            self.nodes.append(child_node)
            self.id += 1
            self.generate_children(child_node)


class FaultTreeGenerator:
    '''
    Generates fault trees:

    current_id = int() next availible unique id.
    max_depth = int() max depth of the generated fault tree
    max_children = int() max children per node in the fault tree
    min_children = int() minimum number of child nodes per fault tree, by default = 2
    '''
    def __init__():
        pass


    # Methods
    def generate_fault_tree(self, current_id, max_depth, max_children, min_children = 2):
        FT = FaultTree(current_id, min_children, max_children, max_depth)
        id = FT.id
        return FT, id

class Converter:
    pass

    def convert_to_graph(FT):
        graph = nx.DiGraph()
        for node in FT.nodes:
            graph.add_node(node.features['id'], **node.features)
        for node in FT.nodes:
            for node_ in node.parents:
                graph.add_edge(node.id, node_.id)
        return graph

class Generator:
    def __init__(self, start_id, max_depth, max_children, min_children, num_models):
        self.start_id = start_id
        self.max_depth = max_depth
        self.max_children = max_children
        self.min_children = min_children
        self.num_models = num_models
        self.graph = nx.DiGraph()
        

    def generate_data(self):
            for i in range(self.num_models):        
                FT, self.start_id = FaultTreeGenerator.generate_fault_tree(self.start_id, self.max_depth, self.max_children, self.min_children)
                new_graph = Converter.convert_to_graph(FT)
                for node, features in new_graph.nodes(data=True):
                    print(f"Node {node}: {features}")
                self.graph = nx.compose(self.graph, new_graph)
            return self.graph 