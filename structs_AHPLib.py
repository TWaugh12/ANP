from AhpAnpLib import functions_AHPLib as reqLib
from AhpAnpLib import ratings_AHPLib as rate
# import functions_AHPLib as reqLib
# import ratings_AHPLib as rate
class Model:
    id_generator = reqLib.itertools.count(0)
    def __init__(self,name=None):
        
        # choose appropriate comparative word that will be saved in the keyword variable. Usually we use: "important","preferred", "likely" can support different keywords
        self.modelID=next(self.id_generator)
        if(name!=None):
            self.name=name
        else:
            self.name="Untitled"
        self.clusters=[]
        # modelType pairwise or ratings 
        self.modelType="pairwise" 
        self.nodeConnections=[]
        self.clusterConnections=[]

        self.all_pc_matrices=[]
        self.all_pr_vectors=[]
        self.pc_with_respect_to=[]
        self.pc_node_order=[]
        self.cluster_neighbors_mtrx=[]
        self.cluster_pc_matrices=[]
        self.supermatrix=[]
        self.weighted_supermatrix=[]
        self.limit=[]
        self.questionnaires=[]
        self.nlabels=[]
        self.clabels=[]
        self.wrtlabels=[]
        self.rateModel=rate.RatStruct(self)
        

        

    def __repr__(self):
        return "\nModel Name: "+self.name+"\nType:"+self.modelType+"\nNodes: "+str(self.getNodeListFromClusters())+"\n Clusters: "+str(self.clusters)+"\n Node Connections: "+str(self.nodeConnections)+"\n Cluster Connections: "+str(self.clusterConnections)
    
    def addCluster2Model(self,cluster):
    #add a cluster that DOES NOT belong to a model to this model
        if isinstance(cluster, (Cluster)):
            if (cluster.parModel==""):
                if cluster not in self.clusters:
                    self.clusters.append(cluster)
                    cluster.parModel=self.modelID
                    # print(cluster.parModel, "--",self.modelID)
            else:
                print("Cluster already assigned to model")
        else:
            print(f"Error: Given object {cluster} not a cluster")
    def addMultipleClusters2Model(self,*clusters):
         for cluster in clusters:
             self.addCluster2Model(cluster)
    
    def remClusterByNameFromModel(self,clusterName):
        cluster=self.getClusterObjByName(clusterName)
    #add a cluster that DOES NOT belong to a model to this model
        # print(cluster.parModel, "--",self.modelID)
        if (cluster.parModel==self.modelID):
            if cluster in self.clusters:
                self.clusters.remove(cluster)
                cluster.parCluster=""

                #   update model connections
                self.defineAllNodeConnections()
                #update cluster connections
                self.defineAllClusterConnections()
        else:
            print("Cluster not assigned to this model")

    def remClusterObjFromModel(self,clusterObj):
        # print(cluster.parModel, "--",self.modelID)
        if (clusterObj.parModel==self.modelID):
            if clusterObj in self.clusters:
                self.clusters.remove(clusterObj)
                clusterObj.parCluster=""

                #   update model connections
                self.defineAllNodeConnections()
                #update cluster connections
                self.defineAllClusterConnections()
        else:
            print("Cluster not assigned to this model")

    def getClusterListFromClusters(self):
        clusters=[]
        for cluster in sorted(self.clusters, key=reqLib.op.attrgetter('order')):
            for node in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')):
                parCluster=node.parCluster
                clusters.append(parCluster)
        return clusters
    def getListNumNodesInClusters(self):
        clusters=[]
        for cluster in sorted(self.clusters, key=reqLib.op.attrgetter('order')):
            tot=0
            for node in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')):
                tot+=1
            clusters.append(tot)
        return clusters
    def getNodeListFromClusters(self):
        nodes=[]
        for cluster in sorted(self.clusters, key=reqLib.op.attrgetter('order')):
            for node in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')):
                nodes.append(node)
        return nodes
    def defineAllNodeConnections(self):
        self.nodeConnections={}
        for cluster in self.clusters:
            for node in cluster.nodes:
                self.nodeConnections[node]=node.connectedTo
        return self.nodeConnections
    
    def getClusterIndexByName(self,name):
        for index, item in enumerate(self.clusters):
            if item.name == name:
                break
        else:
            index = -1
        return index
    
    def getClusterObjByName(self,name):
        for item in self.clusters:
            # print(item.name)
            if item.name == name:
                return item
        return -1
        
    def getClusterObjByID(self,ID):
        for item in self.clusters:
            # print(item.name)
            if item.clusterID == ID:
                return item
        return -1
    
    def getClusterIDByName(self,name):
        for item in self.clusters:
            # print(item.name)
            if item.name == name:
                return item.clusterID
        return -1

    def getNodeIndexInClusterModelByName(self,name):
        indexN = -1
        for index, item in enumerate(self.clusters):
            for indexN, itemN in enumerate(item.nodes):
                if itemN.name == name:
                    return indexN
        return indexN
    
    def getNodeInClusterModelByName(self,name):

        for item in self.clusters:
            for itemN in item.nodes:
                if itemN.name == name:
                    # print(itemN)
                    return itemN
        return -1
    
    def getNodeIDByName(self,name):

        for item in self.clusters:
            for itemN in item.nodes:
                if itemN.name == name:
                    # print(itemN)
                    return itemN.nodeID
        return -1

    def getNodeObjByName(self,name):
        for item in self.clusters:
            for itemN in item.nodes:
                if itemN.name == name:
                    # print(itemN)
                    return itemN
        return -1
    
    def addNodeConnectionFromTo(self,nodeFromName,nodeToName, verb=False):
         #will add a connection from nodeFrom to the nodeTo node

        nodeFrom=self.getNodeInClusterModelByName(nodeFromName)
        # print("From {} ".format(nodeFrom))
        nodeTo=self.getNodeInClusterModelByName(nodeToName)
        # print("To {} ".format(nodeTo))
        if (isinstance(nodeFrom, (Node)) & isinstance(nodeTo, (Node))) :
            if nodeTo not in nodeFrom.connectedTo:
                nodeFrom.connectedTo.append(nodeTo)
            else:
                print("Trying to add n{}. Node already in connections.".format(str(nodeTo.nodeID)))
            if verb==True: 
                print("node: "+nodeFrom.name+" connectedTo: "+str(nodeFrom.connectedTo))

            #   update model connections
            self.defineAllNodeConnections()
            #update cluster connections
            self.defineAllClusterConnections()
        else:
            print("Error: Nodes not found or given objects not nodes in addNodeConnectionFromTo")


    def addNodeConnectionFromNodeToAllNodesOfCluster(self,nodeFromName,clusterToName):
         #will add a connection from nodeFrom to each nodeTo of clusterToName
         clusterTo=self.getClusterObjByName(clusterToName)
         for nodeTo in clusterTo.nodes:
            self.addNodeConnectionFromTo(nodeFromName,nodeTo.name)

    def addNodeConnectionFromAllNodesToAllNodesOfCluster(self,clusterFromName,clusterToName):
         #will add a connection from each nodeFrom of clusterFromName to each nodeTo of clusterToName
        clusterFrom=self.getClusterObjByName(clusterFromName)
        if isinstance(clusterFrom, (Cluster)):
            for nodeFrom in clusterFrom.nodes:
                self.addNodeConnectionFromNodeToAllNodesOfCluster(nodeFrom.name,clusterToName)   
        else:
            print(f"Error: Given object not a cluster in addNodeConnectionFromAllNodesToAllNodesOfCluster")
    def addNodeConnectionFromNodeToNodeList(self,nodeFromName,*nodeNames):
         #will add a connection from  nodeFrom to each nodeTo
        nodeFromObj=self.getNodeInClusterModelByName(nodeFromName)
        if isinstance(nodeFromObj, (Node)):
            for nodeToName in nodeNames:
                nodeToTempObj=self.getNodeInClusterModelByName(nodeToName)
                if isinstance(nodeToTempObj, (Node)):
                    self.addNodeConnectionFromTo(nodeFromName,nodeToName)
                else:
                    print(f"Error: Given object {nodeToTempObj} not a node in addNodeConnectionFromNodeToNodeList")  
        else:
            print(f"Error: Given object {nodeFromObj} not a node in addNodeConnectionFromNodeToNodeList")
    def remNodeConnectionFromTo(self,nodeFromName,nodeToName,verb=False):
        flag=0
        nodeFrom=self.getNodeInClusterModelByName(nodeFromName)
        # print("From {} ".format(nodeFrom))
        nodeTo=self.getNodeInClusterModelByName(nodeToName)
        # print("To {} ".format(nodeTo))

        if nodeTo in nodeFrom.connectedTo:
            nodeFrom.connectedTo.remove(nodeTo)
        else:
            print("Trying to delete n{}. Node not in connections.".format(str(nodeTo.nodeID)))
        if verb==True: 
            print("node: "+nodeFrom.name+" after the deletion is now only connected to: "+str(nodeFrom.connectedTo))

        #   update model connections
        self.defineAllNodeConnections()
        #update cluster connections
        self.defineAllClusterConnections()
   

    def defineAllClusterConnections(self):
        self.clusterConnections={}
        for cluster in self.clusters:
            self.clusterConnections[cluster]=cluster.calcAllConnectionsFrom()
        return self.clusterConnections

    def remAllNodeConnectionsTo(self,nodeName):
        flag=0
        nodeTo_del_id=self.getNodeIDByName(nodeName)
        for nodeFrom in self.getNodeListFromClusters():
            for nodeTo in nodeFrom.connectedTo:
                if nodeTo.nodeID==nodeTo_del_id:
                    flag=1
                    nodeFrom.connectedTo.remove(nodeTo)
                    print("node connection from: "+nodeFrom.name+" to :"+nodeName+" removed.")
                    #update cluster connections
                    nodeTo.parCluster.calcAllConnectionsFrom()
        if flag==0:
            print("Trying to delete n{}. Node was not found.".format(nodeName))
      

    def remAllNodeConnectionsFrom(self,nodeFromName):
        flag=0 
        toDel=[]
        nodeFrom=self.getNodeInClusterModelByName(nodeFromName)
        if(nodeFrom==-1):
            print("Node not found in model - process could not be completed")
        else:
            print(f"Connected to: {nodeFrom.connectedTo}")
            for nodeTo in nodeFrom.connectedTo:
                toDel.append(nodeTo)
            print(*toDel, sep='\n')
            for nodeTo in toDel:
                self.remNodeConnectionFromTo(nodeFrom.name,nodeTo.name)


    def showAllNodeConnectionsFrom(self, nodeFromName):
        flag=0
        nodeFrom_id=self.getNodeIDByName(nodeFromName)
        for nodeFrom in self.getNodeListFromClusters():
            if nodeFrom.nodeID==nodeFrom_id:
                
                for nodeTo in nodeFrom.connectedTo:
                    if flag==0:
                        print("Connections from node "+str(nodeFrom) +" to: ")
                    flag=1
                    print(str(nodeTo))
        if(flag==0):
            print("No connections from",nodeFromName)
    def retAllNodeConnectionsFrom(self, nodeFromName):
        nodelist_connected_to=[]
        for nodeFrom in self.getNodeListFromClusters():
            if(nodeFrom.name==nodeFromName):
                for nodeTo in nodeFrom.connectedTo:
                        nodelist_connected_to.append(nodeTo)
        return nodelist_connected_to



    def showAllNodeConnectionsTo(self, nodeToName):
        #same with enumerating the connectedTo list of each node
        flag=0
        nodeTo_id=self.getNodeIDByName(nodeToName)
        for nodeFrom in self.getNodeListFromClusters():
            for nodeTo in nodeFrom.connectedTo:
                if nodeTo.nodeID==nodeTo_id:
                    flag=1
                    print("Connections from node "+str(nodeFrom) +" to: "+str(nodeTo))        

        if(flag==0):
            print("No connections to",nodeToName)

    def showAllNodeConnections(self):
        #same with enumerating the connectedTo list of each node
        for nodeFrom in self.getNodeListFromClusters():
            flag=0
            for nodeTo in nodeFrom.connectedTo:
                    if(flag==0):
                        print("Connections from node "+str(nodeFrom))
                    flag=1
                    print(" to: "+str(nodeTo))
            if(flag==0):
                print("No connections from",nodeFrom.name)

    def showAllClusterConnections(self):
        #same with enumerating the connectedTo list of each node
        for clusterFrom in self.clusters:
            flag=0
            for clusterTo in clusterFrom.connectedTo:
                    if(flag==0):
                        print("Connection(s) from cluster "+str(clusterFrom))
                    flag=1
                    print(" to: "+str(clusterTo))
            if(flag==0):
                print("No connections from",clusterFrom.name)  

    def retAllClusterConnectionsFrom(self,clusterName,verb=False):
        #same with enumerating the connectedTo list of each node
        connected_to=[]
        clusterFrom_id=self.getClusterIDByName(clusterName)
        for clusterFrom in self.clusters:
            flag=0
            if clusterFrom.clusterID==clusterFrom_id:
                if(flag==0):
                    if(verb):
                        print("Connection(s) from cluster "+str(clusterFrom.name))
                    flag=1
                inner_flag=0
                for clusterTo in clusterFrom.connectedTo:
                    connected_to.append(clusterTo)
                    inner_flag=1
                    if(verb):
                        print(" to: "+str(clusterTo.name))
                if(inner_flag==0):
                    if(verb):
                        print("No connections from",clusterFrom.name)
                    return None
                     
        if(verb):       
            print(f"Results list:{connected_to}")
        return connected_to 

    def retAllClusterConnectionsFromNode(self,nodeName,verb=False):
        #to which clusters is this node connected to
        connected_to=[]
        flag=0
        node_id=self.getNodeIDByName(nodeName)

        # print("working with",node_id)
        for nodeFrom in self.getNodeListFromClusters():
            if nodeFrom.nodeID==node_id:
                for nodeTo in nodeFrom.connectedTo:
                    flag=1
                    clusterTo=nodeTo.parCluster
                    if(clusterTo not in connected_to):
                        if (verb):
                            print("Connections from node "+str(nodeFrom.name) +" to cluster: "+str(clusterTo.name)) 
                        connected_to.append(clusterTo)      
        if(flag==0):
             if verb==True:
                print("No connections from",nodeName)
        return connected_to
    
    def showAllClusterConnectionsFrom(self,clusterName):
        #same with enumerating the connectedTo list of each node
        clusterFrom_id=self.getClusterIDByName(clusterName)
        for clusterFrom in self.clusters:
            flag=0
            if clusterFrom.clusterID==clusterFrom_id:
                for clusterTo in clusterFrom.connectedTo:
                        if(flag==0):
                            print("Connection(s) from cluster "+str(clusterFrom))
                            flag=1
                        print(" to: "+str(clusterTo))
                if(flag==0):
                    print("No connections from",clusterFrom.name) 
        
    def listAllClusterConnections(self,clusterFrom):
        connected_to=[]
        #same with enumerating the connectedTo list of each node
        
        for clusterTo in clusterFrom.connectedTo:
            connected_to.append(clusterTo)
        return connected_to       
    
   
       
    def moveNode2Cluster(self,node,toCluster,withConnections,verb=False):    
        #moving node from Cluster A to Cluster B (toCluster) of a specific model
        my_parCluster=node.parCluster
        parID=self.getClusterObjByID(my_parCluster.clusterID)
        if(verb):
            print(f"Initially we have parent cluster {my_parCluster.name} of node {node.name}")
        if(parID!=-1):
            my_cl_indx=self.getClusterIndexByName(my_parCluster.name)
            if(my_cl_indx!=-1):
                self.clusters[my_cl_indx].nodes.remove(node)
                if(verb):
                    print(f"Now my cluster has the following nodes:{self.clusters[my_cl_indx].nodes}")
            dest_cl_indx=self.getClusterIndexByName(toCluster.name)
            if(dest_cl_indx!=-1):
                self.clusters[dest_cl_indx].nodes.append(node)
                self.clusters[dest_cl_indx].nodes[-1].parCluster=dest_cl_indx
                if(withConnections==False):
                    self.clusters[dest_cl_indx].nodes[-1].connectedTo=[]
                    #update cluster connections
                self.defineAllClusterConnections()
                if(verb):
                    print(f"My destination cluster has the following nodes:{self.clusters[dest_cl_indx].nodes}")
                    print(f"My node's parent now is :{self.clusters[dest_cl_indx].nodes[-1].parCluster}")
                    print(f"My node's connections To now are :{self.clusters[dest_cl_indx].nodes[-1].connectedTo}")
    def getMaxNodeOrder(self):
        max=-1
        for cluster in self.clusters:
            for node in cluster.nodes:
             if(node.order>max):
                 max=node.order
        return max
    def getMaxClusterOrder(self):
        max=-1
        for cluster in self.clusters:
            if(cluster.order>max):
                max=cluster.order
        return max
    def printStruct(self):
        print("_________________________MODEL STRUCTURE_________________________")
        
        print("Name: "+self.name)
        print("Type: "+self.modelType)
        print("\n____________________________NODES_______________________________")
        for this_node in self.getNodeListFromClusters():
            print(this_node)
        print("____________________________CLUSTERS____________________________")
        for this_cluster in self.clusters:
            print(this_cluster)
        print("_________________________NODE CONNECTIONS___________________________")
        self.showAllNodeConnections()
        print("_________________________CLUSTER CONNECTIONS__________________")
        self.showAllClusterConnections()
        if(self.modelType=="ratings"):
            print("_________________________RATING MODEL SET UP__________________")
            print(self.rateModel)

    def setModelTypeRatings(self):
        self.modelType="ratings"
    def drawGraphNodes(self):
        G = reqLib.nx.DiGraph()

        # Add nodes with cluster assignments
        node_colors = []
        color_mapping = {}  # Dictionary to store color assignments
        for clusterObj in self.clusters:
            for nodeObj in clusterObj.nodes:
                cluster_id = nodeObj.parCluster.clusterID
                G.add_node(nodeObj, cluster=cluster_id, name=nodeObj.name)
                if cluster_id not in color_mapping:
                # Assign a new color for the cluster
                    color_mapping[cluster_id] = len(color_mapping)
                node_colors.append(color_mapping[cluster_id])
        # print(f"colors{color_mapping} nodes {node_colors}")

        # Add edges 
        for clusterObj in self.clusters:
            for nodeObj in clusterObj.nodes:
                nodes_connected_to=self.retAllNodeConnectionsFrom(nodeObj.name)
                # print(f"for {nodeObj} we have {nodes_connected_to}")
                for node_to in nodes_connected_to:
                    if(len(nodes_connected_to)>0):
                        # print(f"From {nodeObj.name} to {node_to.name}")
                        G.add_edge(nodeObj, node_to) 

        # Generate the positions of the nodes
        pos = reqLib.nx.circular_layout(G)
        # pos = reqLib.nx.shell_layout(G)
        
        # Set the size of the initial graph
        reqLib.plt.figure(figsize=(18, 9)) 
        
        # Draw nodes with colors based on clusters
        node_size=8000
        nodes=reqLib.nx.draw_networkx_nodes(G, 
                                            pos, 
                                            node_color=node_colors, 
                                            cmap="rainbow", 
                                            alpha=0.5,
                                            node_shape=">",
                                            node_size=node_size)

        # Draw edges (if you have edges in your graph)
        # reqLib.nx.draw_networkx_edges(G, pos)
        reqLib.nx.draw_networkx_edges(
        G,
        pos,
        # connectionstyle="arc3,rad=0.1",  # Add this parameter to create curved edges
        arrows=True,  # Add arrows to the edges
        )
        # Add labels to nodes
        labels = reqLib.nx.get_node_attributes(G, "name")
        reqLib.nx.draw_networkx_labels(G, pos, labels)

        # Create a legend
        unique_clusters = sorted(color_mapping.keys())
        legend_handles = []
        legend_labels = []
        color_mapping_values = list(color_mapping.values())
        cmap = reqLib.plt.cm.get_cmap("rainbow", len(color_mapping_values))
        for cluster, color_value in zip(unique_clusters, color_mapping_values):
            my_color = cmap(color_value)
            handle = reqLib.plt.Line2D([], [], marker="o", markersize=10, color=my_color)
            legend_handles.append(handle)
            legend_labels.append(f" {self.getClusterObjByID(cluster).name}")
        reqLib.plt.legend(legend_handles, legend_labels, title="Clusters")
        # Display the graph
        reqLib.plt.axis("off")
        reqLib.plt.show()

    def drawGraphClusters(self):
        G = reqLib.nx.DiGraph()

        # Add clusters 

        for clusterObj in self.clusters:
                G.add_node(clusterObj.name, name=clusterObj.name)
                

        # Add edges 
        for clusterObj in self.clusters:
            clusters_connected_to=self.retAllClusterConnectionsFrom(clusterObj.name)
            print(f"Cluster from {clusterObj.name} connected to: {clusters_connected_to}")
            if(clusters_connected_to!=None):
                    for cluster_to in clusters_connected_to:
                            # print(f"From {nodeObj.name} to {node_to.name}")
                            G.add_edge(clusterObj.name, cluster_to.name) 

        # Generate the positions of the nodes
        pos = reqLib.nx.circular_layout(G)
        # pos = reqLib.nx.shell_layout(G)
        
        # Set the size of the initial graph
        reqLib.plt.figure(figsize=(18, 9)) 
        
        # Draw nodes with colors based on clusters
        node_size=8000
        nodes=reqLib.nx.draw_networkx_nodes(G, 
                                            pos, 
                                            cmap="rainbow", 
                                            alpha=0.5,
                                            node_shape="s",
                                            node_size=node_size)

        # Draw edges (if you have edges in your graph)
        # reqLib.nx.draw_networkx_edges(G, pos)
        reqLib.nx.draw_networkx_edges(
        G,
        pos,
        # connectionstyle="arc3,rad=0.1",  # Add this parameter to create curved edges
        arrows=True,  # Add arrows to the edges
        )
        # Add labels to nodes
        labels = reqLib.nx.get_node_attributes(G, "name")
        reqLib.nx.draw_networkx_labels(G, pos, labels)

        # # Create a legend
        # unique_clusters = sorted(color_mapping.keys())
        # legend_handles = []
        # legend_labels = []
        # color_mapping_values = list(color_mapping.values())
        # cmap = reqLib.plt.cm.get_cmap("rainbow", len(color_mapping_values))
        # for cluster, color_value in zip(unique_clusters, color_mapping_values):
        #     my_color = cmap(color_value)
        #     handle = reqLib.plt.Line2D([], [], marker="o", markersize=10, color=my_color)
        #     legend_handles.append(handle)
        #     legend_labels.append(f" {self.getClusterObjByID(cluster).name}")
        # reqLib.plt.legend(legend_handles, legend_labels, title="Clusters")
        # Display the graph
        reqLib.plt.axis("off")
        reqLib.plt.show()

    def drawGraphModel(self,filepath=None, verb=False):
        
        colors = ['#f57a64', '#64b4f5', '#63c273', '#f5e50c', '#cb7bed', '#f7952d', '#f79ee4', '#9ef7e7']

        # Create the main graph
        g = reqLib.gr.Digraph('G', filename='hierarchy.gv')

        # Create the clusters
        for i,cluster in enumerate(sorted( self.clusters, key=reqLib.op.attrgetter('order'))):
            with g.subgraph(name=f'cluster_{i}') as c:
                c.attr(style='filled', color='lightgrey')
                my_color=colors[i%8]
                for nodeObj in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')):
                    c.node(nodeObj.name, color=my_color, style='filled')  # additional nodes without edges
            c.attr(label=cluster.name)

        

        # Create connections between the nodes in the clusters
   
        # g.edge('a1', 'b3')
        for clusterObj in self.clusters:
            for node_from in clusterObj.nodes:
                nodes_connected_to = self.retAllNodeConnectionsFrom(node_from.name)
                if len(nodes_connected_to) > 0:
                    for node_to in nodes_connected_to:
                        g.edge(node_from.name, node_to.name)

        # Save the graph to a PNG file
        g.format = 'png'
        g.render(self.name)

        if filepath:
            if reqLib.os.path.isfile(filepath):
                # If the Excel file exists, load it
                workbook = reqLib.openpyxl.Workbook()
                workbook = reqLib.openpyxl.load_workbook(filepath)

                # Create a new worksheet named "Graph Model" and insert the PNG image
                worksheet = workbook.create_sheet("Graph Model")
                img = reqLib.Image(self.name + ".png")

                # Save to Excel
                worksheet.add_image(img, "A1")
                workbook.save(filepath)

                if verb:
                    print("PNG image appended to the existing Excel file.")
            else:
                # If the Excel file doesn't exist, create a new one
                workbook = reqLib.openpyxl.Workbook()

                # Create a new worksheet named "Graph Model" and insert the PNG image
                worksheet = workbook.active
                worksheet.title = "Graph Model"
                img = reqLib.Image(self.name + ".png")

                # Save to Excel
                worksheet.add_image(img, "A1")
                workbook.save(filepath)

                if verb:
                    print("PNG image saved to a new Excel file.")
        else:
            if verb:
                print("PNG image saved to your working folder.")



      
class Node:
    id_generator = reqLib.itertools.count(0)

    def __init__(self, node_name,node_order):
        self.name = node_name
        self.order = node_order
        self.nodeID = next(self.id_generator)
        self.parCluster=""
        self.connectedTo=[]

    def __repr__(self):
        return self.name+" order: "+ str(self.order)
        #return self.name+" NID: "+str(self.nodeID)+" order: "+ str(self.order)
    def myParentCluster(self):
        return self.parCluster
    def updateN_DisplayOrder(self,newOrder):
        self.order=newOrder

class Cluster:
    id_generator = reqLib.itertools.count(0)
    
    def __init__(self, cluster_name,cluster_order):
        self.name = cluster_name
        self.order = cluster_order
        self.clusterID = next(self.id_generator)
        self.nodes=[]
        self.connectedTo=[]
        self.parModel=""
       
    def __repr__(self):
        return self.name+" order: "+ str(self.order)
        # return self.name+" CID:  "+str(self.clusterID)+" order: "+ str(self.order)
    def printWithNodes(self):
        print("Cluster: {} with nodes: {}\n".format(self.name,self.nodes))
    def addNode2Cluster(self,node):
    #add a node that DOES NOT belong to a cluster to this cluster
        if isinstance(node, (Node)):
            if (node.parCluster==""):
                if node not in self.nodes:
                    self.nodes.append(node)
                    node.parCluster=self
            else:
                print("Node already assigned to cluster")
        else:
            print(f"Error: Given object {node} not a node")
    def addMultipleNodes2Cluster(self,*nodes):
         for node in nodes:
             self.addNode2Cluster(node)

    # def remNodeFromCluster(self,node):
    # #rem a node from the cluster -- not to be used if node has been assigned to model

    #     # print(self.clusterID, "--",node.parCluster.clusterID,"--",node.parCluster.parModel)
    #     if (node.parCluster.parModel==""):
    #         if (node.parCluster.clusterID==self.clusterID):
    #             if node not in self.nodes:
    #                 self.nodes.remove(node)
    #                 node.parCluster=""
    #         else:
    #             print("Node not assigned to this cluster")
    #     else: 
    #             print("Node already assigned to model - cannot be removed with this command. Remove it from the model by deleting the line corresponding to the assignment of the node to the model ")
    def forceAddNode2Cluster(self,node):
    #add a node even if it already belongs to another cluster to a this cluster
        ClusterA=node.parCluster
        ClusterA.nodes.remove(node)
        if node not in self.nodes:
                self.nodes.append(node)
                node.parCluster=self

    # def detachFromModel(self):
    #     self.parModel=""

    def calcAllConnectionsFrom(self,verb=False):
        # print("cluster connections for: "+self.name)
        # print('cluster nodes: {}'.format(self.nodes))
        for fromNode in self.nodes:
            # print("From node:",str(fromNode.nodeID))
            for toNode in fromNode.connectedTo:
                # print("To node:",str(toNode.nodeID))
                if toNode.parCluster not in self.connectedTo:
                    self.connectedTo.append(toNode.parCluster)
                    if verb==True:
                        print("connecting from cluster: {} to cluster {}".format(str(self.clusterID),str(toNode.parCluster.clusterID)))
                # else:
                    # print("Connection from cluster: {} to cluster {} already exists".format(str(self.clusterID),str(toNode.parCluster.clusterID)))
        return self.connectedTo
    def updateC_DisplayOrder(self,newOrder):
        self.order=newOrder

    def getMaxNodeOrderOfCluster(self):
        max=-1
        for node in self.nodes:
            if(node.order>max):
                max=node.order
        return max

