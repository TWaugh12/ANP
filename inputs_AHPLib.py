from AhpAnpLib import functions_AHPLib as reqLib
from AhpAnpLib import calcs_AHPLib as cdf_calc
from AhpAnpLib import ratings_AHPLib as cdf_rat
from AhpAnpLib import structs_AHPLib as cdf_str
# import functions_AHPLib as reqLib
# import calcs_AHPLib as cdf_calc
# import ratings_AHPLib as cdf_rat
# import structs_AHPLib as cdf_str

def readStructFromExcel(model,filepath,sheet_name, verbal=False):
    print("Initiating process to import model structure from Excel file") 

    all_excel = reqLib.pd.read_excel (filepath, sheet_name=sheet_name, header=None)
    array_excel=all_excel.fillna(0)
    num_rows=array_excel.shape[0]
    num_columns = array_excel.shape[1]
    count=0
    if(verbal==True):
        print("File Len=", num_rows)
        print("Num of Cols=", num_columns)
        print("excel",array_excel)
    for i in range(0,num_rows):
        cluster=array_excel.iloc[i,0]
        cluster_obj=cdf_str.Cluster(cluster,i)

        if(verbal==True):
            print("Cluster:", cluster)

        for j in range(1,num_columns):
            if(array_excel.iloc[i,j]!=0):
                node=array_excel.iloc[i,j]
                node_obj=cdf_str.Node(node,count)
                count+=1
                cluster_obj.addNode2Cluster(node_obj)

                if(verbal==True):
                    print("Node:", node)

        model.addCluster2Model(cluster_obj)
    print("Structure imported from Excel file") 
        
def readConnectionsFromExcel(model,filepath,sheet_name, verbal=False):  
    print("Initiating process to import connections from Excel file") 

    all_excel = reqLib.pd.read_excel (filepath, sheet_name=sheet_name, header=None)
    array_excel=all_excel.fillna(0)
    num_rows=array_excel.shape[0]
    num_columns = array_excel.shape[1]
    error=" "
    if(verbal==True):
        print("File Len=", num_rows)
        print("Num of Cols=", num_columns)
        print("excel",array_excel)
    # for each column find column label
    # for each node From 
    # check full column if 1 then read row label and check if node, if yes then add connection otherwise add to error text
    for j in range(1,num_columns):
        elem_from_nm=array_excel.iloc[0,j]
        from_obj=model.getNodeIDByName(elem_from_nm)

        if (from_obj!=-1):
            for i in range(1,num_rows):
                elem_to_nm=array_excel.iloc[i,0]
                to_obj=model.getNodeIDByName(elem_to_nm)
                if (to_obj!=-1):
                    elem_connect=array_excel.iloc[i,j]
                    if(elem_connect!=0 and elem_connect!=1):
                        error+="\nError: Unexpected value in cell"+i+","+j
                    elif (elem_connect==1):
                        model.addNodeConnectionFromTo(elem_from_nm,elem_to_nm, verbal)        
                else:
                    error+="\nError: Could not find a node with name "+elem_to_nm+" in row "+str(i)
        else:
            error+="\nError: Could not find a node with name "+elem_from_nm+" in column "+str(j)
    if (error!=" "):
        print(error)
    else:
        print("Connections imported from Excel file") 

def readSupermatrixFromExcel(model,filepath,sheet_name, verbal=False):  
    print("Initiating process to import struct and values from the supermatrix given in the Excel file") 

    all_excel = reqLib.pd.read_excel (filepath, sheet_name=sheet_name, header=None)
    array_excel=all_excel.fillna(0)
    num_rows=array_excel.shape[0]
    num_columns = array_excel.shape[1]
    current_cl=-1 
    error=" "
    if(verbal==True):
        print("File Len=", num_rows)
        print("Num of Cols=", num_columns)
        print("excel",array_excel)
    #read struct from col 0 and 1 of the excel sheet   
    for i in range(2,num_rows):
        cluster_name=array_excel.iloc[i,0]
        who_am_i=model.getClusterObjByName(cluster_name)
        if(who_am_i==-1 and cluster_name!=0):
            cluster_obj=cdf_str.Cluster(cluster_name,i)
            current_cl=cluster_obj
            if(verbal==True):
                print("Adding Cluster:", cluster_name)
        node_name=array_excel.iloc[i,1]
        who_am_i_nd=model.getNodeObjByName(node_name)
        if(who_am_i_nd==-1 and node_name!=0):
            node_obj=cdf_str.Node(node_name,i)
            current_cl.addNode2Cluster(node_obj)
            if(verbal==True):
                print("Adding Node:", node_name)
        model.addCluster2Model(current_cl)
        
    #validate that supermatrix has the same labels in rows and columns
    for j in range(2,num_columns):
        cluster_name=array_excel.iloc[0,j]
        who_am_i=model.getClusterObjByName(cluster_name)
        if(who_am_i==-1 and cluster_name!=0):
            error+="\n Warning: "+cluster_name+ " in col "+str(j+1)+" does not correspond to the cluster name in row "+str(j+1)
        node_name=array_excel.iloc[1,j]
        who_am_i_nd=model.getNodeIDByName(node_name)
        if(who_am_i_nd==-1 and node_name!=0):
            error+="\n Warning: "+node_name+ " in col "+str(j+1)+" does not correspond to the node name in row "+str(j+1) 
    
    #define connections 
    # for each column find column label and then for each node From 
    # check full column if 1 then read row label and check if node, if yes then add connection otherwise add to error text
    dir_vector=[]
    for j in range(2,num_columns):
        elem_from_nm=array_excel.iloc[1,j]
        from_obj=model.getNodeIDByName(elem_from_nm)
        if (from_obj!=-1):
            for i in range(2,num_rows):
                elem_to_nm=array_excel.iloc[i,1]
                to_id=model.getNodeIDByName(elem_to_nm)
                to_obj=model.getNodeObjByName(elem_to_nm)
                if (to_id!=-1):
                    elem_connect=array_excel.iloc[i,j]
                    if (elem_connect!=0):
                        dir_vector.append(elem_connect)
                        model.addNodeConnectionFromTo(elem_from_nm,elem_to_nm, verbal)
                        size=len(to_obj.parCluster.nodes)
                        if (verbal==True):
                            print("size:",size)
                            print("dir_vector:",dir_vector)
                        if(len(dir_vector)==size):
                                pc_matrix = reqLib.np.empty([size, size], dtype = float)
                                pc_matrix=convDirect2PairwiseMatrix(dir_vector,verbal)
                                model.all_pc_matrices.append(pc_matrix)
                                dir_vector=[]
                else:
                    error+="\nError: Could not find a node with name "+elem_to_nm+" in row "+str(i)
        else:
            error+="\nError: Could not find a node with name "+elem_from_nm+" in column "+str(j)
    if (verbal==True):
        print("values read from excel:")
        print( model.all_pc_matrices)
   
    if (error!=" "):
        print(error)
    else:
        print("Supermatrix imported from Excel file. Note that messages like: \'Cluster already assigned to model\' are expected when a cluster has more than one nodes, since it will be created only once.") 

def genFullQuest(model,keyword,verb=False):
    questnr=[]
    for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 

        for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
            
            connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
            for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                i=0
                qset=[]
                # print("Nodes:\n",clusterPWC.nodes)
                for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                    j=0
                    # print("NodeA: ",nodeA.nodeID)
                    for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        # print("NodeB: ",nodeB.nodeID)
                        if (nodeA.nodeID!= nodeB.nodeID) and (i<j) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                            # print("{}-{}".format(i,j))
                            quest="With respect to {}, which one is more {}: {} or {} ? By how much?".format(nodeFrom.name, keyword, nodeA.name, nodeB.name)
                            if verb:
                                print(quest)
                            qset.append(quest)
                        j+=1
                    i+=1
                questnr.append(qset)
                if verb:
                    print("---------------------------------------------------------------------------\n")
    return questnr
def genFirstLineAboveDiagQuest(model,keyword,verb=False):
    questnr=[]
    for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 

        for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
            
            connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
            for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                i=0
                qset=[]
                # print("Nodes:\n",clusterPWC.nodes)
                for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                    j=0
                    # print("NodeA: ",nodeA.name)
                    for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        # print("NodeB: ",nodeB.name)
                        # print("{}-{}".format(i,j),nodeA in nodeFrom.connectedTo,nodeB in nodeFrom.connectedTo)
                        if (nodeA.nodeID!= nodeB.nodeID) and ((i==0)or (i+1==j)) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                            
                            quest="With respect to {}, which one is more {}: {} or {} ? By how much?".format(nodeFrom.name, keyword, nodeA.name, nodeB.name)
                            if verb:
                                print(quest)
                            qset.append(quest)
                        j+=1
                    i+=1
                questnr.append(qset)
                if verb:
                    print("---------------------------------------------------------------------------\n")
    return questnr
def genFirstLineQuest(model,keyword,verb=False):
    questnr=[]
    for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 

        for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
            
            connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
            for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                i=0
                qset=[]
                # print("Nodes:\n",clusterPWC.nodes)
                nodeA=clusterPWC.nodes[0]
                j=0
                # print("NodeA: ",nodeA.name)
                for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                    # print("NodeB: ",nodeB.name)
                    # print("{}-{}".format(i,j),nodeA in nodeFrom.connectedTo,nodeB in nodeFrom.connectedTo)
                    if (nodeA.nodeID!= nodeB.nodeID) and (i<j) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                        # print("{}-{}".format(i,j))
                        quest="With respect to {}, which one is more {}: {} or {} ? By how much?".format(nodeFrom.name, keyword, nodeA.name, nodeB.name)
                        if verb:
                            print(quest)
                        qset.append(quest)
                    j+=1
                questnr.append(qset)
                if verb:
                 print("---------------------------------------------------------------------------\n")
                
    return questnr

def genexport4QualtricsQuestFull(filepath,model,keyword,howMuch=True):
    count=1
    with open(filepath, 'w', encoding='UTF8',newline='') as f: 
        f.write("[[AdvancedFormat]]\n")
        for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 

            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
                f.write("[[Block: With respect to: "+nodeFrom.name+"]]\n")
                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                    i=0

                    # print("Nodes:\n",clusterPWC.nodes)
                    for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        j=0
                        # print("NodeA: ",nodeA.nodeID)
                        for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                            # print("NodeB: ",nodeB.nodeID)
                            if (nodeA.nodeID!= nodeB.nodeID) and (i<j) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                                # print("{}-{}".format(i,j))
                            
                                f.write("\n[[Question:MC:SingleAnswer]]\n")
                                quest1_csv=str(count)+'. With respect to '+nodeFrom.name+ ' which one is more '+keyword+':'+'\n'
                                f.write(quest1_csv) 
                                f.write("\n[[Choices]]\n")
                                
                                ans1=nodeA.name+'\n'
                                f.write(ans1)
                                ans2=nodeB.name+'\n'
                                f.write(ans2)
                                count+=1
                                
                                if(howMuch):
                                    f.write("\n[[Question:MC:Dropdown]]\n")
                                    quest2_csv=str(count)+". By how much?"+'\n'
                                    f.write(quest2_csv)
                                    f.write("\n[[Choices]]\n")
                                    f.write("1\n2\n3\n4\n5\n6\n7\n8\n9\n")
                                    count+=1
                                f.write('\n')
                                
                            j+=1
                        i+=1
def genexport4QualtricsFirstLineAboveDiagQuest(filepath,model,keyword,howMuch=True):
    count=1
    with open(filepath, 'w', encoding='UTF8',newline='') as f:
        f.write("[[AdvancedFormat]]\n")

        for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 

            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
                f.write("[[Block: With respect to: "+nodeFrom.name+"]]\n")
                
                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                    i=0
                    # f.write("[[Block:"+clusterPWC.name+"]]\n")
                    # print("Nodes:\n",clusterPWC.nodes)
                    for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        j=0
                        # print("NodeA: ",nodeA.nodeID)
                        for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                            # print("NodeB: ",nodeB.nodeID)
                            if (nodeA.nodeID!= nodeB.nodeID) and ((i==0)or (i+1==j)) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                                f.write("\n[[Question:MC:SingleAnswer]]\n")
                                quest1_csv=str(count)+'. With respect to '+nodeFrom.name+ ' which one is more '+keyword+':'+'\n'
                                f.write(quest1_csv) 
                                f.write("\n[[Choices]]\n")
                                
                                ans1=nodeA.name+'\n'
                                f.write(ans1)
                                ans2=nodeB.name+'\n'
                                f.write(ans2)
                                count+=1
                                
                                if(howMuch):
                                    f.write("\n[[Question:MC:Dropdown]]\n")
                                    quest2_csv=str(count)+". By how much?"+'\n'
                                    f.write(quest2_csv)
                                    f.write("\n[[Choices]]\n")
                                    f.write("1\n2\n3\n4\n5\n6\n7\n8\n9\n")
                                    count+=1
                                f.write('\n')
                            j+=1
                        i+=1       
def genexport4QualtricsFirstLineQuest(filepath,model,keyword,howMuch=True):
    count=1
    with open(filepath, 'w', encoding='UTF8',newline='') as f:
        f.write("[[AdvancedFormat]]\n")
        for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 
            
            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
                f.write("[[Block: With respect to: "+nodeFrom.name+"]]\n")
                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                    i=0
                    qset=[]
                    # print("Nodes:\n",clusterPWC.nodes)
                    nodeA=clusterPWC.nodes[0]
                    j=0
                    # print("NodeA: ",nodeA.nodeID)
                    for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        # print("NodeB: ",nodeB.nodeID)
                        if (nodeA.nodeID!= nodeB.nodeID) and (i<j) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                                f.write("\n[[Question:MC:SingleAnswer]]\n")
                                quest1_csv=str(count)+'. With respect to '+nodeFrom.name+ ' which one is more '+keyword+':'+'\n'
                                f.write(quest1_csv) 
                                f.write("\n[[Choices]]\n")
                                
                                ans1=nodeA.name+'\n'
                                f.write(ans1)
                                ans2=nodeB.name+'\n'
                                f.write(ans2)
                                count+=1
                                
                                if(howMuch):
                                    f.write("\n[[Question:MC:Dropdown]]\n")
                                    quest2_csv=str(count)+". By how much?"+'\n'
                                    f.write(quest2_csv)
                                    f.write("\n[[Choices]]\n")
                                    f.write("1\n2\n3\n4\n5\n6\n7\n8\n9\n")
                                    count+=1
                                f.write('\n')
                        j+=1

def genexport4GoogleQuestFull(filepath,model,keyword,howMuch=True):
    count=1
    with open(filepath, 'w', encoding='UTF8',newline='') as f:
        writer = reqLib.csv.writer(f)  
        for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 

            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
                
                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                    i=0

                    # print("Nodes:\n",clusterPWC.nodes)
                    for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        j=0
                        # print("NodeA: ",nodeA.nodeID)
                        for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                            # print("NodeB: ",nodeB.nodeID)
                            if (nodeA.nodeID!= nodeB.nodeID) and (i<j) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                                # print("{}-{}".format(i,j))
                                quest1_csv=["With respect to "+nodeFrom.name+' which one is more '+keyword, nodeA.name, nodeB.name]
                                writer.writerow(quest1_csv)  
                                quest2_csv=["By how much?","1","2","3","4","5","6","7","8","9"]
                                writer.writerow(quest2_csv)                            
                            j+=1
                        i+=1
def genexport4GoogleFirstLineAboveDiagQuest(filepath,model,keyword,howMuch=True):
    count=1
    with open(filepath, 'w', encoding='UTF8',newline='') as f:
        writer = reqLib.csv.writer(f)  

        for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 

            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
                
                
                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                    i=0
                    # f.write("[[Block:"+clusterPWC.name+"]]\n")
                    # print("Nodes:\n",clusterPWC.nodes)
                    for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        j=0
                        # print("NodeA: ",nodeA.nodeID)
                        for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                            # print("NodeB: ",nodeB.nodeID)
                            if (nodeA.nodeID!= nodeB.nodeID) and ((i==0)or (i+1==j)) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                                quest1_csv=["With respect to "+nodeFrom.name+' which one is more '+keyword, nodeA.name, nodeB.name]
                                writer.writerow(quest1_csv)  
                                quest2_csv=["By how much?","1","2","3","4","5","6","7","8","9"]
                                writer.writerow(quest2_csv)  
                            j+=1
                        i+=1       
def genexport4GoogleFirstLineQuest(filepath,model,keyword,howMuch=True):
    count=1
    with open(filepath, 'w', encoding='UTF8',newline='') as f:
        writer = reqLib.csv.writer(f)  
        for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')): 
            
            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
               
                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                    i=0
                    qset=[]
                    # print("Nodes:\n",clusterPWC.nodes)
                    nodeA=clusterPWC.nodes[0]
                    j=0
                    # print("NodeA: ",nodeA.nodeID)
                    for nodeB in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        # print("NodeB: ",nodeB.nodeID)
                        if (nodeA.nodeID!= nodeB.nodeID) and (i<j) and (nodeA in nodeFrom.connectedTo) and (nodeB in nodeFrom.connectedTo):
                                quest1_csv=["With respect to "+nodeFrom.name+' which one is more '+keyword, nodeA.name, nodeB.name]
                                writer.writerow(quest1_csv)  
                                quest2_csv=["By how much?","1","2","3","4","5","6","7","8","9"]
                                writer.writerow(quest2_csv)  
                        j+=1

def export4ExcelQuestFull(model,filepath,show_estimates=False,verb=False):
    try:
        workbook = reqLib.xlsxwriter.Workbook(filepath)
        worksheet = workbook.add_worksheet("pairwise_comp")
        worksheet.set_column(0,20,15)

        cell_format_hdTitle  = workbook.add_format({'bold': True,'font_color': 'red', 'font_size':16})
        cell_format_hdTitle.set_pattern(1)
        cell_format_hdTitle.set_bg_color('C3D5FF')
        cell_format_hd  = workbook.add_format({'bold': True,'font_color': 'red'})
        cell_format_hd.set_pattern(1)
        cell_format_hd.set_bg_color('C3D5FF')
        cell_format_hd2  = workbook.add_format({'bold': True,'font_color': 'red'})
        cell_format_hd2.set_pattern(1)
        cell_format_hd2.set_bg_color('#d5ffc3')
        cell_format_hd3  = workbook.add_format({'bold': True,'font_color': 'red'})
        
        
        cell_format_ln  = workbook.add_format({'bold': True,'font_color': 'blue'})
        cell_format_ln.set_border(1)
        cell_format_diag=workbook.add_format()
        cell_format_diag.set_pattern(1)
        cell_format_diag.set_border(1)
        cell_format_diag.set_bg_color('FFFF6B')
        cell_format_block=workbook.add_format()
        cell_format_block.set_pattern(1)
        cell_format_block.set_border(1)
        cell_format_block.set_bg_color('#808080')
        cell_format_dir=workbook.add_format()
        cell_format_dir.set_pattern(1)
        cell_format_dir.set_border(1)
        cell_format_dir.set_bg_color('#ceebd6')
        cell_format_empty  = workbook.add_format()
        cell_format_empty.set_border(1)
        
        decimal_format = workbook.add_format({'num_format': '0.00','bg_color': '#accbe8',  'border': '1'})
        res_hd_format = workbook.add_format({'bold': True,'font_color': 'blue','bg_color': '#accbe8',  'border': '1'})
        inc_hd_format=workbook.add_format({'bold': True,'font_color': 'orange','bg_color': '#f9d5b6',  'border': '1'})
        inc_decimal_format = workbook.add_format({'num_format': '0.00','bg_color': '#f9d5b6',  'border': '1'})
        row=0
        col=0
        for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')):
            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 

                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                if(len(connectedClusters)>0):
                    worksheet.write(row, 0, nodeFrom.name,cell_format_hdTitle)
                    if(verb):
                        print(row,nodeFrom.name)
                    row=row+1
                    for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                        if(verb):
                            print(row,"Enter judgments:\n",row+1,clusterPWC.name)
                        worksheet.write(row, 0, "Enter pairwise comparisons in the white cells of the table or numerical data in the green cells. For the Direct Values column, if the smallest value is best, invert the value before entering it (e.g., $10 as =1/10) .",cell_format_hd3)
                        worksheet.write(row+1, 0, clusterPWC.name,cell_format_hd2)
                        row=row+1
                        col=0
                        size=0
                        if(verb):
                            print(row,"Now nodes\n")
                        for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                            # print("NodeA: ",nodeA.nodeID)         
                            
                            if(nodeA in nodeFrom.connectedTo):
                                worksheet.write(row, col+1, nodeA.name,cell_format_ln)
                                worksheet.write(row+col+1, 0, nodeA.name,cell_format_ln)
                                worksheet.write_number(row+col+1, col+1,1.000 ,cell_format_diag)
                                # print(nodeA.name)
                                col+=1
                                size+=1
                            worksheet.write(row, size+1, "Direct values",cell_format_ln)
                            # if(verb):
                            #     print(row,"Direct")
                        for a in range(size):
                            for b in range (size):
                                if(a!=b and a>b):
                                    val='=1/'+convIndex2ExcelRef(row+b+2,a+2)
                                    worksheet.write(row+a+1,b+1,val,cell_format_block)
                                if(a!=b and a<b):
                                    worksheet.write(row+a+1,b+1,"",cell_format_empty)
                            worksheet.write(row+a+1,size+1,"",cell_format_dir)
                        if(show_estimates==True):
                            worksheet.write(row,col+2,"Line Sum",res_hd_format)
                            worksheet.write(row,col+3,"Estimated Priority",res_hd_format)
                            worksheet.write(row+size+1,0,"Sum of Col",inc_hd_format)
                            worksheet.write(row+size+2,col+2,"Est. Incons.",inc_hd_format)
                            for a in range(size):
                                val='='
                                valColSum=''
                                for b in range (size):
                                    val+='+'+convIndex2ExcelRef(row+a+2,b+2)
                                    valColSum='=sum('+convIndex2ExcelRef(row+2,b+2)+':'+convIndex2ExcelRef(row+2+size-1,b+2)+')'
                                    worksheet.write(row+size+1,b+1,valColSum,inc_decimal_format)
                                worksheet.write(row+a+1,col+2,val,decimal_format)
                                
                            valRowSum='=sum('+convIndex2ExcelRef(row+2,col+3)+':'+convIndex2ExcelRef(row+2+size-1,col+3)+')'

                            worksheet.write(row+a+2,col+2,valRowSum,decimal_format)
                            
                            #calc tem priorities
                            for a in range(size):
                                val='='+convIndex2ExcelRef(row+a+2,col+3)+'/'+convIndex2ExcelRef(row+size+2,col+3)
                                worksheet.write(row+a+1,col+3,val,decimal_format)
                            #calc inconsistency
                            #ci=(lmax-size)/(size-1)
                            priorRef=convIndex2ExcelRef(row+2,col+4)+':'+convIndex2ExcelRef(row+2+size-1,col+4)
                            colSumRef=convIndex2ExcelRef(row+size+2,2)+':'+convIndex2ExcelRef(row+size+2,size+1)
                            valInc='=((MMULT('+colSumRef+','+priorRef+')-'+str(size)+')/('+str(size)+'-1))/'+str(cdf_calc.RI(size))
                            worksheet.write(row+size+2,col+3,valInc,inc_decimal_format)
                        # if verb:
                        #     print(nodeA.name,size)        
                        
                        row=row+col+3
                        if verb:
                            print("Saved pairwise comparison matrix for cluster : ",clusterPWC.name, "with respect to node:", nodeFrom.name)
                            print(f"Next row: {row}")    
                    # row+=1
                    if verb:
                            print(f"Row {row} {cluster.name} {nodeFrom.name}")   
        workbook.close()
    except Exception as e:
        print(f"An error occured while trying to save the pairwise comparison matrices to the file. Check if the file is already open in Excel and if yes, then please close it and retry.\n Error: {e}")
def export4ExcelRatingScales(model,filepath,load_values=True, show_estimates=False, verb=False):
    # print(verb)
    if(verb):
        print("setting up rating scales comparison matrices ")
    border = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    # border_noright = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style=None), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    # border_noleft = reqLib.Border(left=reqLib.Side(style=None), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    lower_diag_fill = reqLib.PatternFill(start_color="bcbcbc", end_color="bcbcbc", fill_type="solid")
    res_hd_fill = reqLib.PatternFill(start_color="ffb6e5f2", end_color="ffb6e5f2", fill_type="solid")
    dir_hd_fill = reqLib.PatternFill(start_color="6aa84f", end_color="6aa84f", fill_type="solid")
    dir_fill = reqLib.PatternFill(start_color="b6d7a8", end_color="b6d7a8", fill_type="solid")
    
    diag_light = reqLib.PatternFill(start_color="FFFF6B", end_color="FFFF6B", fill_type="solid")
    incons = reqLib.PatternFill(start_color="f9d5b6", end_color="f9d5b6", fill_type="solid")
    estimates= reqLib.PatternFill(start_color="accbe8", end_color="accbe8", fill_type="solid")
    bold_font=reqLib.Font(bold=True)
    # rule_n = reqLib.DataBarRule(start_type='min', start_value=0, end_type='max', end_value=0,
    #                color=reqLib.Color(rgb='fff2463a'), showValue=True)
    # rule_i = reqLib.DataBarRule(start_type='min', start_value=0, end_type='max', end_value=0,
    #                color=reqLib.Color(rgb='ff0b30b5'), showValue=True)
    
    worksheetName="rating_scales"
    try:
        if reqLib.os.path.isfile(filepath):
            #print("load")
            workbook = reqLib.openpyxl.load_workbook(filepath)
        else:
            #print("create")
            workbook = reqLib.openpyxl.Workbook()
            if 'Sheet' in workbook.sheetnames:
                del workbook['Sheet']
            if worksheetName not in workbook.sheetnames:
                workbook.create_sheet(title=worksheetName)

        workbook.active = workbook.sheetnames.index(worksheetName)
        worksheet = workbook.active
          
        row=1
        col=1
        if(model.modelType=="ratings"):
            for my_crit in model.rateModel.ratCriteria:
                #the criterion name as a title
                worksheet.cell(row, col).value= my_crit.name
                worksheet.cell(row, col).fill = res_hd_fill
                worksheet.cell(row, col).font = bold_font
                hd_row=row
                row+=1
                my_scale=model.rateModel.getScaleOfCriterion(my_crit)
                if(verb):
                    print(f"Criterion: {my_crit}")
                    print(my_scale)
                # print("Error!!!?",my_scale,my_crit)
                # print(my_scale.members)
                size=0
                #the scale matrix
                for item in my_scale.members:
                    worksheet.cell(hd_row, col+1).fill = res_hd_fill

                    worksheet.cell(row, col+1).value= item[0]
                    worksheet.cell(row, col+1).border = border
                    worksheet.cell(row, col+1).font = bold_font
                    
                    worksheet.cell(row+col,1).value= item[0]
                    worksheet.cell(row+col,1).border = border
                    worksheet.cell(row+col,1).font = bold_font
                    col+=1
                    size+=1
                
                # direct values
                worksheet.cell(hd_row, col+1).fill = res_hd_fill
                worksheet.cell(row, col+1).fill =dir_hd_fill
                worksheet.cell(row, col+1).value ="Direct values"
                worksheet.cell(row, col+1).border = border
                worksheet.cell(row, col+1).font = bold_font
             
                #inside the table
                for inside_i in range(size+1):
                    worksheet.cell(row+inside_i, col+1).fill = dir_fill
                    worksheet.cell(row+inside_i, col+1).border = border
                    for inside_j in range(size+1):
                        worksheet.cell(row+inside_i, inside_j+1).border = border
                        if(load_values==True & (inside_i>0)&(inside_j>0)&(inside_i<inside_j)):
                            worksheet.cell(row+inside_i, inside_j+1).value=my_scale.val_mat[inside_i-1,inside_j-1]
                            worksheet.cell(row+inside_i, inside_j+1).number_format = '0.000'
                        if((inside_i==inside_j) &(inside_i>0)):
                            worksheet.cell(row+inside_i, inside_j+1).fill=diag_light
                            worksheet.cell(row+inside_i, inside_j+1).value=1.0
                        if((inside_i>inside_j) &(inside_i>0)):
                            worksheet.cell(row+inside_i, inside_j+2).fill=lower_diag_fill
                            # worksheet.cell(row+inside_i, inside_j+2).protection = reqLib.Protection(locked=True)
                            worksheet.cell(row+inside_i, inside_j+2).value='=1/'+convIndex2ExcelRef(row+inside_j+1,inside_i+1)
                            worksheet.cell(row+inside_i, inside_j+2).number_format = '0.000'
                #show estimates
                if(show_estimates==True):
                    set_cell_value_and_style(worksheet.cell(row,col+2),"Line Sum",estimates,border, "ff123ef1" ,True)
                    set_cell_value_and_style(worksheet.cell(row,col+3),"Est. Normal Priorities",estimates,border,"ff123ef1",True)
                    set_cell_value_and_style(worksheet.cell(row,col+4),"Est. Ideal Priorities",estimates,border,"ff123ef1",True)
                    set_cell_value_and_style(worksheet.cell(row+size+1,1),"Sum of Col",incons,border,"ff000000",True)
                    set_cell_value_and_style(worksheet.cell(row+size+2,col+2),"Est. Incons.",incons,border,"ff000000",True)

                    for a in range(size):
                        val='='
                        valColSum=''
                        for b in range (size):
                            val+='+'+convIndex2ExcelRef(row+a+1,b+2)
                            valColSum='=sum('+convIndex2ExcelRef(row+1,b+2)+':'+convIndex2ExcelRef(row+size,b+2)+')'
                            set_cell_value_and_style(worksheet.cell(row+size+1,b+2),valColSum,incons,border, "ff000000" ,True)
                            worksheet.cell(row+size+1,b+2).number_format = '0.000'
                        set_cell_value_and_style(worksheet.cell(row+a+1,col+2),val,estimates,border, "ff123ef1" ,False)
                        worksheet.cell(row+a+1,col+2).number_format = '0.000'
                    valRowSum='=sum('+convIndex2ExcelRef(row+1,col+2)+':'+convIndex2ExcelRef(row+size,col+2)+')'
                    set_cell_value_and_style(worksheet.cell(row+a+2,col+2),valRowSum,estimates,border, "ff123ef1" ,False)
                    worksheet.cell(row+a+2,col+2).number_format = '0.000'
                    #calc tem priorities
                    for a in range(size):
                        val='='+convIndex2ExcelRef(row+a+1,col+2)+'/'+convIndex2ExcelRef(row+size+1,col+2)
                        ideal_val='='+convIndex2ExcelRef(row+a+1,col+3)+'/MAX('+convIndex2ExcelRef(row+1,col+3)+':'+convIndex2ExcelRef(row+size+1,col+3)+')'
                        set_cell_value_and_style(worksheet.cell(row+a+1,col+3),val,estimates,border, "ff123ef1" ,True)
                        set_cell_value_and_style(worksheet.cell(row+a+1,col+4),ideal_val,estimates,border, "ff123ef1" ,True)
                        worksheet.cell(row+a+1,col+3).number_format = '0.000'
                        worksheet.cell(row+a+1,col+4).number_format = '0.000'
                    #calc inconsistency
                    #ci=(lmax-size)/(size-1)
                    priorRef=convIndex2ExcelRef(row+1,col+3)+':'+convIndex2ExcelRef(row+size,col+3)
                    colSumRef=convIndex2ExcelRef(row+size+1,2)+':'+convIndex2ExcelRef(row+size+1,size+1)
                    valInc='=((MMULT('+colSumRef+','+priorRef+')-'+str(size)+')/('+str(size)+'-1))/'+str(cdf_calc.RI(size))
                    set_cell_value_and_style(worksheet.cell(row+size+2,col+3),valInc,incons,border, "ff000000" ,True)
                    worksheet.cell(row+size+2,col+3).number_format = '0.000'
                #end of show estimates
                row=row+size+4
                col=1
            for column in worksheet.columns:
                column_letter = column[0].column_letter
                worksheet.column_dimensions[column_letter].width = 20
            workbook.save(filepath)
 
    except Exception as e:
        print(f"An error occured while trying to save the rating scale tables to the file. Check if the file is already open in Excel and if yes, then please close it and retry.\n Error: {e}")
def export4ExcelRatingsTable(model,filepath,verb=False):

    worksheetName="rating_table"
    
    bold_font=reqLib.Font(bold=True)
    res_hd_fill = reqLib.PatternFill(start_color="ffb6e5f2", end_color="ffb6e5f2", fill_type="solid")
    res_cr_fill = reqLib.PatternFill(start_color="ffcde4f7",fill_type="solid")
    tot_res_fill = reqLib.PatternFill(start_color="ffffffcc",fill_type="solid")
    tot_hd_fill = reqLib.PatternFill(start_color="fff2fa07",fill_type="solid")
    pr_res_fill = reqLib.PatternFill(start_color="ffffffb3",fill_type="solid")
    border = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    no_border = reqLib.Border()
    incons = reqLib.PatternFill(start_color="f9d5b6", end_color="f9d5b6", fill_type="solid")
    results_fill = reqLib.PatternFill(start_color="b6d7a8", end_color="b6d7a8", fill_type="solid")
    try:
         if reqLib.os.path.isfile(filepath):
             print("load")
             workbook = reqLib.openpyxl.load_workbook(filepath)
         else:
             print("create")
             workbook = reqLib.openpyxl.Workbook()
            #  if 'Sheet' in workbook.sheetnames:
            #      del workbook['Sheet']
         if worksheetName not in workbook.sheetnames:
            workbook.create_sheet(title=worksheetName)

         workbook.active = workbook.sheetnames.index(worksheetName)
         worksheet = workbook.active
        
         row=1
         col=1
         if(model.modelType=="ratings"):
             #rows: alternatives
             #cols: sel. criteria
             #cells: scale of selected criterion
             prev_end=-3
             for ind,my_crit in enumerate(model.rateModel.ratCriteria):
                 # print(ind,my_crit)
                 #this is empty
                 # print(model.rateModel.ratCritPriorities)
                 len_rat_tbl=3+len(model.rateModel.ratAlternatives)
                 col+=1
                 #the criterion name as a title
                 set_cell_value_and_style(worksheet.cell(row,col), my_crit.name,res_hd_fill,border, "ff000000" ,True)
                 worksheet.cell(row+len_rat_tbl+2,col).value=model.rateModel.ratCritPriorities[ind]
                 #"=INDEX(rating_scales!"+convIndex2ExcelRef(ref_from,col_res)+":"+convIndex2ExcelRef(ref_to,col_res)+", MATCH("+convIndex2ExcelRef(in_row,col)+",rating_scales!"+c_from+":"+c_to+", 0))"
                 worksheet.cell(row+len_rat_tbl+2,col).border = border
                 worksheet.cell(row+len_rat_tbl+2,col).number_format = '0.000'
                
                 #add scale cells
                 my_scale=model.rateModel.getScaleOfCriterion(my_crit)
                 ref_from=6+prev_end
                 c_from="A"+str(ref_from)
                 ref_to=6+prev_end+len(my_scale.members)-1
                 c_to="A"+str(ref_to)
                 
                 prev_end=6+prev_end+len(my_scale.members)-1
                 # print(f"from:{c_from}-ti:{c_to}-prev:{prev_end}")
                 in_row=2
                 
                 col_res=len(my_scale.members)+5
                 for my_alt in model.rateModel.ratAlternatives:
                     worksheet=addDropDown2ExcelCell(worksheet,c_from,c_to,"rating_scales",convIndex2ExcelRef(in_row,col))
                     worksheet.cell(in_row,col).fill = incons
                     worksheet.cell(in_row,col).border = border

                     # excel did not recognize xlookup and vlookup worksheet.cell(in_row+len_rat_tbl+2,col).value= "=VLOOKUP("+convIndex2ExcelRef(in_row,col)+",rating_scales!"+c_from+":"+convIndex2ExcelRef(ref_to,col_res)+","+str(col_res)+")"
                     worksheet.cell(in_row+len_rat_tbl+3,col).value= "=INDEX(rating_scales!"+convIndex2ExcelRef(ref_from,col_res)+":"+convIndex2ExcelRef(ref_to,col_res)+", MATCH("+convIndex2ExcelRef(in_row,col)+",rating_scales!"+c_from+":"+c_to+", 0))"
                     worksheet.cell(in_row+len_rat_tbl+3,col).border = border
                     worksheet.cell(in_row+len_rat_tbl+3,col).number_format = '0.000'
                     in_row+=1
                 set_cell_value_and_style(worksheet.cell(in_row+2,col), "",results_fill,no_border, "ff000000" ,True)
                 #the criteria priorities
                #  set_cell_value_and_style(worksheet.cell(in_row+2,col), "value",res_cr_fill,border, "ff000000" ,True)
                 #the criterion name as a title
                 set_cell_value_and_style(worksheet.cell(in_row+5,col), my_crit.name,res_hd_fill,border, "ff000000" ,True)
             col=1
             row+=1   
             num_crit=len(model.rateModel.ratCriteria)
             num_alts=len(model.rateModel.ratAlternatives)
             row_start=row
             for my_alt in model.rateModel.ratAlternatives:
                 #row heading tbl 1
                 set_cell_value_and_style(worksheet.cell(row,col), my_alt.name,res_hd_fill,border, "ff000000" ,True)
                 #row heading tbl 2
                 set_cell_value_and_style(worksheet.cell(row+len_rat_tbl+3,col), my_alt.name,res_hd_fill,border, "ff000000" ,True)
                 #results totals (estimates)
                 valRowSum='=sumproduct('+convIndex2ExcelRef(row+len_rat_tbl+3,col+1)+':'+convIndex2ExcelRef(row+len_rat_tbl+3,col+num_crit)+','+convIndex2ExcelRef(row_start+len_rat_tbl+1,col+1)+':'+convIndex2ExcelRef(row_start+len_rat_tbl+1,col+num_crit)+')'
                 set_cell_value_and_style(worksheet.cell(row+len_rat_tbl+3,col+num_crit+1),valRowSum,tot_res_fill,border, "ff123ef1" ,False)
                 worksheet.cell(row+len_rat_tbl+3,col+num_crit+1).number_format = '0.000'
                
                 #results totals (priorities)
                 
                 valRowPr='='+convIndex2ExcelRef(row+len_rat_tbl+3,col+num_crit+1)+'/sum('+convIndex2ExcelRef(row_start+len_rat_tbl+3,col+num_crit+1)+':'+convIndex2ExcelRef(row_start+len_rat_tbl+2+num_alts,col+num_crit+1)+')'
                 set_cell_value_and_style(worksheet.cell(row+len_rat_tbl+3,col+num_crit+2),valRowPr,pr_res_fill,border, "ff000000" ,False)
                 worksheet.cell(row+len_rat_tbl+3,col+num_crit+2).number_format = '0.000'
                 row+=1
             #write results table
             # =XLOOKUP(B2,rating_scales!$A$3:$A$6,rating_scales!$H$3:$H$6)    
             col=1
             row+=2
             set_cell_value_and_style(worksheet.cell(row,col), "ESTIMATED TOTALS AND PRIORITIES",results_fill,no_border, "ff000000" ,True)
             set_cell_value_and_style(worksheet.cell(row+2,col+num_crit+1), "TOTALS",tot_hd_fill,border, "ff000000" ,True)
             set_cell_value_and_style(worksheet.cell(row+2,col+num_crit+2), "PRIORITIES",tot_hd_fill,border, "ff000000" ,True)
             set_cell_value_and_style(worksheet.cell(in_row+2,col+num_crit+1), "",results_fill,no_border, "ff000000" ,True)
             set_cell_value_and_style(worksheet.cell(in_row+2,col+num_crit+2), "",results_fill,no_border, "ff000000" ,True)
             row+=1
            
         for column in worksheet.columns:
             column_letter = column[0].column_letter
             worksheet.column_dimensions[column_letter].width = 20
         workbook.save(filepath)
            
    
    
    except Exception as e:
        print(f"An error occured while trying to save the rating table to the file. Check if the file is already open in Excel and if yes, then please close it and retry.\n Error: {e}")

def export4ExcelRatingsSetup(model,exportfile,show_estimates=False,verb=False):
    cdf_calc.calcRateCritV(model,verbal=verb)
    export4ExcelRatingScales(model,exportfile,True,show_estimates,False)
    export4ExcelRatingsTable(model,exportfile,False)
    cdf_calc.updateRateTblResults2Excel(model,exportfile,"rating_table",verb=verb)

def calcExcelRatings(model,importfile,exportfile,verb=False):
    importRatingsFromExcel(model,importfile,"rating_scales",verb)    
    importRatTableFromExcel(model,importfile,"rating_table",verb)  
    calcRatResults(model,exportfile,verb)

def addLabels2Excel(label_list,filepath,wsheet_name,start_row,start_col,direction):
    #direction 0 horizontal, 1 vertical, 2 both
    bold_font=reqLib.Font(bold=True)
    res_hd_fill = reqLib.PatternFill(start_color="ffb6e5f2", end_color="ffb6e5f2", fill_type="solid")
    border = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    no_border = reqLib.Border()
    
    try:
        if reqLib.os.path.isfile(filepath):
            # print("load")
            workbook = reqLib.openpyxl.load_workbook(filepath)
            worksheet = workbook[wsheet_name]
            
            #worksheet = workbook.active
            row=start_row
            col=start_col
            for i,label in enumerate(label_list):
                if(direction==0):
                    if(label!=label_list[i-1]):
                        set_cell_value_and_style(worksheet.cell(row+i,col), label,res_hd_fill,border, "ff000000" ,True)
                    else:
                        set_cell_value_and_style(worksheet.cell(row+i,col), "",res_hd_fill,no_border, "ff000000" ,True)
                if(direction==1):
                    if(label!=label_list[i-1]):
                        set_cell_value_and_style(worksheet.cell(row,col+i), label,res_hd_fill,border, "ff000000" ,True)
                    else:
                        set_cell_value_and_style(worksheet.cell(row,col+i), "",res_hd_fill,no_border, "ff000000" ,True)
                # print(f"label{label},row{row+i}, col{col+i}\n")
                       
            for column in worksheet.columns:
                column_letter = column[0].column_letter
                worksheet.column_dimensions[column_letter].width = 15
            max_col_dir=3
            if (direction==0):
                max_col_dir=len(label_list)+2
            for row in worksheet.iter_rows(min_row=1, min_col=2, max_row=len(label_list)+2, max_col=max_col_dir):
                for cell in row:
                    if cell.value is not None:
                        cell.border = border
            workbook.save(filepath)
    except Exception as e:
        print(f"An error occured while trying to save the cluster labelsto the file. Check if the file is already open in Excel and if yes, then please close it and retry.\n Error: {e}")

def importFromExcel(model,filepath,sheet_name, verbal=False):

    all_excel = reqLib.pd.read_excel (filepath, sheet_name=sheet_name, header=None)
    array_excel=all_excel.fillna(0)
    file_len=len(array_excel)
    if(verbal==True):
        print("File Len=", file_len)
        print("excel",array_excel)
    row=0
    for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')):
        if(verbal==True): print("Cluster:",cluster.name)
        for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
            connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)

            if(len(connectedClusters)>0):
                if(verbal==True): print("NFrom:",nodeFrom.name)
                if(verbal==True): print("Connected clusters",connectedClusters)
                model.wrtlabels.append(array_excel.iat[row,0])
                for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                    row+=2
                    if(verbal==True): print("From file row: ",row,"working in cluster:",array_excel.iat[row,0],clusterPWC.name)
                    model.clabels.append(clusterPWC.name) #not in excel anymore
                    row=row+1
                    size=0
                    for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                        if(nodeA in nodeFrom.connectedTo):
                            model.nlabels.append(array_excel.iat[row+size,0])
                            size+=1

                        if(verbal==True): print("Current row",row,"All node labels till this point:",model.nlabels)
                    pc_matrix =  reqLib.np.empty([size, size], dtype = float)
                    #check if there are direct entries, then take them to generate the matrix otherwise use the pw judgments
                    dir_count=0
                    dir_vector=reqLib.np.empty([size], dtype = float)
                    for dir_val in range(size):
                        dir_item=array_excel.iat[dir_val+row,size+1]
                        if(dir_item>0):
                            dir_vector[dir_count]=dir_item
                            dir_count+=1
                            if(verbal):
                                print("-->dir-->",dir_item)
                    if(dir_count==size):
                        if(verbal):
                            print("will not read judgments")
                        pc_matrix=convDirect2PairwiseMatrix(dir_vector,verbal)  
                    else:
                        for a in range(size):
                            for b in range (size):
                                item=array_excel.iat[a+row,b+1]
                                # print("item",item)
                                if a==b:
                                    pc_matrix[a,b]=1.0
                                elif a<b:
                                    pc_matrix[a,b]=item
                                    if(item!=0):
                                        pc_matrix[b,a]=1./item
                                    else:
                                        pc_matrix[b,a]=0
                    if(verbal==True): print(pc_matrix)
                    model.all_pc_matrices.append(pc_matrix)
                    row=row+size+1
                row+=1

def importRatingsFromExcel(model,filepath,sheet_name, verbal=False):

    all_excel = reqLib.pd.read_excel (filepath, sheet_name=sheet_name, header=None) 
    array_excel=all_excel.fillna(0)
    file_len=len(array_excel)
    # if(verbal==True):
    #     print("File Len=", file_len)
    #     print("excel",array_excel)
    row=0
    if(model.modelType=="ratings"):
        while(row<file_len):
            my_crit_name=array_excel.iat[row,0]
            my_crit=model.rateModel.getRatCritObjByName(my_crit_name)
            
            if (verbal):
                print(f"row: {row}--for criterion {my_crit_name}")

            my_scale=model.rateModel.getScaleOfCriterion(my_crit)
            if (verbal):
                print(f"scale {my_scale.name}")
            row+=2
            size=len(my_scale.members)
            
            # for each criterion read a members by members matrix and save it for calculations
            
            pc_matrix =  reqLib.np.empty([size, size], dtype = float)
            #check if there are direct entries, then take them to generate the matrix otherwise use the pw judgments
            dir_count=0
            dir_vector=reqLib.np.empty([size], dtype = float)
            for dir_val in range(size):
                dir_item=array_excel.iat[dir_val+row,size+1]
                if(dir_item>0):
                    dir_vector[dir_count]=dir_item
                    dir_count+=1
                    if(verbal):
                        print("-->dir-->",dir_item)
            if(dir_count==size):
                if(verbal):
                    print("will not read judgments")
                pc_matrix=convDirect2PairwiseMatrix(dir_vector,verbal)  
            else:    
                for a in range(size):
                    for b in range (size):
                        item=array_excel.iat[a+row,b+1]
                        # print("item",item)
                        if a==b:
                            pc_matrix[a,b]=1.0
                        elif a<b:
                            pc_matrix[a,b]=item
                            if(item!=0):
                                pc_matrix[b,a]=1./item
                            else:
                                pc_matrix[b,a]=0
            if(verbal==True):
                print(pc_matrix)
          
            my_val_mat=reqLib.cp.deepcopy(pc_matrix)
            model.rateModel.updateScaleMatrix(my_scale.name,my_val_mat.copy())
            row+=size+3       
    else:
        print("Cannot read ratings if model is not set up as a ratings model")
    
def importRatTableFromExcel(model,filepath,sheet_name, verbal=False):

    all_excel = reqLib.pd.read_excel (filepath, sheet_name=sheet_name, header=None) 
    array_excel=all_excel.fillna(0)
    file_len=len(array_excel)
    # if(verbal==True):
    #     print("File Len=", file_len)
    #     print("excel",array_excel)
    row=1
    n_alt=len(model.rateModel.ratAlternatives)
    n_crit=len(model.rateModel.ratCriteria)
    r_matrix = reqLib.np.full((n_alt, n_crit), -1)
    if(model.modelType=="ratings"):
        for i,alt in enumerate(model.rateModel.ratAlternatives):
            col=0
            for j,crit in enumerate(model.rateModel.ratCriteria):
                value=array_excel.iat[row+i,col+j+1]
                scale=model.rateModel.getScaleOfCriterion(crit)
                
                priority=scale.getmValueBymName(value,verbal)
                index=scale.getmIndexBymName(value,verbal)

                r_matrix[i,j]=index

                if(verbal==True):
                    print(f"Reading for crit: {crit.name} and alt: {alt.name}------{value}------>{scale.name} ---{priority} ---> at loc {index}")
            if(verbal==True):
                print("------\n")
        
        model.rateModel.ratMatrix=reqLib.cp.deepcopy(r_matrix)
        if(verbal==True):
            print(reqLib.tabulate(r_matrix.tolist(), tablefmt="grid"))
        
def calcRatResults(model,exportfile,verbal=False):
    cdf_calc.calcRateCritV(model,verbal=verbal)
    export4ExcelRatingScales(model,exportfile,True,False,verb=verbal)
    cdf_calc.appendScaleResults2Excel(model,exportfile,"rating_scales", normalbar=False,idealbar=True,verb=verbal)
    export4ExcelRatingsTable(model,exportfile,verbal)
    cdf_calc.appendRateTblResults2Excel(model,exportfile,"rating_table", normalbar=False,idealbar=True,verb=verbal)
def convDirect2PairwiseMatrix(vector,verbal=False):
    size=len(vector)
    pc_matrix =  reqLib.np.empty([size, size], dtype = float)
    for i in range(0,size):
        for j in range(0,size):
            pc_matrix[i,j]=vector[i]/vector[j]
            # print("\n",i,vector[i],j,vector[j],pc_matrix[i,j])
    if(verbal==True):
        print("Vector:",vector,"\nPC Matrix:\n",pc_matrix)
    return pc_matrix    

def readRatScaleRCPfile(scaleName,fileName,verbal=False):
    # read a superdecisions generated scale file and return the vector of member names and judgment pc matrix
    with open(fileName, 'r') as file:
        file_contents = file.read()
    raw_lines = file_contents.split('\n')
    file.close()
    names=[]
    judgments=[]
    values=[]
    memberText="ratings newCategory -network $net -criteria $crit -group $group -category {"
    valueText="set values {"
    for line in raw_lines:
        if( memberText in line):
            info = line.split(memberText)[1].strip("}")
            names.append(info)
        if( valueText in line):
            info = line.split(valueText)[1].strip("}")
            judgments=info.split(",")[0]
            input_list = judgments.split()

    # Convert each element to a float using a list comprehension
    fl_judg_list = [float(elem) for elem in input_list]
    nnodes=len(names)
    matrix_judg=cdf_calc.vector2matrix(fl_judg_list,nnodes)
    priorities=cdf_calc.priorityVector(matrix_judg)
    
    indx=0
    for name in names:
        values.append([name,priorities[indx]])
        indx+=1
    scale=cdf_rat.RatScale(scaleName)
    scale.defineScaleByValue(matrix_judg,False,*values)

    if(verbal):
        print("Reading Judgments from file:"+fileName)
        print(names)
        print(fl_judg_list)
        print("nodes:",nnodes)
        print(matrix_judg)
        print(priorities)
    return scale

def readSDMODfile(modelName,fileName,export=False,verbal=False):
    new_model=cdf_str.Model(modelName)

    with open(fileName, 'r') as file:
        file_contents = file.read()
    raw_lines = file_contents.split('\n')
    file.close()
    netText="set net ["
    structureText="c-network readAllCompares -network $net -source {"
    structureStart=0
    cntNd=0
    totNd=0
    cntCl=0
    clusters=[]
    nodes=[]
    node_pwc_order=[]
    temp_all_pc_matrices=[]
    if (export):
        f_op=open(modelName+'.py', 'w')
        f_op.write('from AhpAnpLib import inputs_AHPLib as input\nfrom AhpAnpLib import structs_AHPLib as str\nfrom AhpAnpLib import calcs_AHPLib as calc\nfrom AhpAnpLib import ratings_AHPLib as rate\nfrom AhpAnpLib import functions_AHPLib as reqLib\n')
        f_op.write('\nnew_model=str.Model("'+modelName+'")\n')
    #find starting point with interesting data
    for i,line in enumerate(raw_lines):
        if( netText in line):
            #tbd create multiple model objects for multilevel nets
            break
        else:
            if(structureText in line):
                structureStart=i+1
                cntCl=int(raw_lines[structureStart])
            line=structureStart+1
    #read model's  clusters and nodes
    for i in range(0,cntCl):
        component=raw_lines[line]
        line+=1
        clusters.append(component) 
        cl=cdf_str.Cluster(component,i)
        if(export):
            f_op.write('\ncl'+str(i)+'=str.Cluster("'+component+'",'+str(i)+')')
        cntNd=int(raw_lines[line])
        line+=1

        for j in range(0,cntNd):
            component=raw_lines[line]
            line+=1
            nodes.append(component)
            nd=cdf_str.Node(component,j+totNd)
            cl.addNode2Cluster(nd)
            if(export):
                f_op.write('\nnd'+str(i)+'=str.Node("'+component+'",'+str(j+totNd)+')')
                f_op.write('\ncl'+str(i)+'.addNode2Cluster(nd'+str(i)+')')
        totNd+=cntNd
        new_model.addCluster2Model(cl)
        if(export):
            f_op.write('\nnew_model.addCluster2Model(cl'+str(i)+')\n')
    #now add cluster info
    line+=1
    component=raw_lines[line]
    info=reqLib.re.split(r'[,\s]+', component)
    int_info_list = [int(elem) for elem in info if elem.strip()]
    # print(int_info_list)
    matSize=int_info_list[0]
    del int_info_list[0]

    cl_neighbors_labels=[]
    for i,item in enumerate(int_info_list):
        if((i%2)==0):
            cur_cl=clusters[item]
        else:
            cl_neighbors_labels.insert(item,cur_cl)
    line+=1
    cl_neigh=[]
    for clLine in range(0,matSize):
        component=raw_lines[line]
        judgments=component.split(" ")
        cl_neighbors_list = [float(elem) for elem in judgments if elem.strip()]
        cl_neigh.append(cl_neighbors_list)
        line+=1
        
    cl_neighbors_df = reqLib.pd.DataFrame(cl_neigh,cl_neighbors_labels,cl_neighbors_labels)  
    new_model.cluster_neighbors_mtrx=cl_neighbors_df

    #for each cluster read cluster matrix
    # print(cntCl)
    for clMatCnt in range(cntCl):
        if('f_op' in locals()):
            cl_df,cur_id,line=sdmodLines2PwMatrix(new_model,raw_lines,line,f_op,verbal)
        else:
            cl_df,cur_id,line=sdmodLines2PwMatrix(new_model,raw_lines,line,None,verbal)
        #nodes section
    #find how many node pwc matrices you will need to read
    component=raw_lines[line]
    temp_ln=line
    count_pw=0
    while component!="}":
        if(component==""):
            count_pw+=1
        # print(component)
        temp_ln+=1
        component=raw_lines[temp_ln]
    
    # print(count_pw)
    #import node pairwise comparisons
    for ndMatCount in range(count_pw):
        if('f_op' in locals()):
            nd_df,cur_id,line=sdmodLines2PwMatrix(new_model,raw_lines,line,f_op,verbal)
        else:
            nd_df,cur_id,line=sdmodLines2PwMatrix(new_model,raw_lines,line,None,verbal)
        if(nd_df.empty==False):
            temp_all_pc_matrices.append(nd_df.to_numpy())
            node_pwc_order.append(cur_id)
    # print(node_pwc_order)
    #re-order to match internal order
    new_model.all_pc_matrices=reqLib.cp.deepcopy(reorder_pwc_np(new_model,temp_all_pc_matrices,node_pwc_order))
    # print("2nd round")
    
    input_file=new_model.name+"_empty.xlsx"
    input_file_full=new_model.name+"_filledIn.xlsx"
    output_file=new_model.name+"_results.xlsx"

    export4ExcelQuestFull(new_model,input_file,show_estimates=False,verb=False)
    if (verbal):
        print("\npc matrices------- ")
        print(new_model.all_pc_matrices)
    cdf_calc.calcAHPMatricesSave2File(new_model,input_file,output_file,inputFileUse=False,normalbar=False,idealbar=True,verbal=False)
    #add the inputs to an input file to easily modify
    cdf_calc.copyExcelSheet(output_file, input_file_full, "pairwise_comp",False)
    # save pairwise comparison matrices 

    if(verbal):
        # print(raw_lines)
        # print(structureStart)
        print(cntCl)
        print(clusters)
        print(nodes)
        print(cl_neighbors_df)
        new_model.printStruct()
    if(export):
        f_op.write('\nnew_model.printStruct()')
        f_op.write('\ninput.export4ExcelQuestFull(new_model,"py_file_'+input_file+'",verb=False)')
        f_op.write('\ncalc.calcAHPMatricesSave2File(new_model,"'+input_file_full+'","py_file_'+output_file+'",inputFileUse=True,normalbar=False,idealbar=True,verbal=False)')
        f_op.write('\ncalc.copyExcelSheet("py_file_'+output_file+'","py_file_'+input_file_full+'", "pairwise_comp",False)')
        f_op.close()    
    return new_model

def getRealPosOfNode(my_order,my_nd_id):
    node_ids=[]
    pos_indx=[]
    for index in range(0, len(my_order), 2):
        node_ids.append(my_order[index])
        pos_indx.append(my_order[index+1])
    # print(node_ids)
    # print(pos_indx)
    is_ordered = node_ids == sorted(node_ids)
    # print("ordered", is_ordered)
    if(is_ordered!=True):
        for item in node_ids:
            if(item==my_nd_id):
                cur = item
                pos=node_ids.index(cur)
                new_pos=pos_indx[pos]
                return new_pos
    return -1

def convertJudgVec2Pairwise(my_list,my_order,mat_size):
    count=0
    print("List",my_list)
    print("Order",my_order)
    my_df=reqLib.np.zeros([mat_size, mat_size], dtype = float)
    init_df=reqLib.np.zeros([mat_size, mat_size], dtype = float)

    for i in range(mat_size):
        for j in range(mat_size):
            if(i==j):
                init_df[i,j]=1.
                my_df[i,j]=1.
            if(i<j):
                init_df[i,j]=my_list[count]
                if(my_list[count]!=0):
                    init_df[j,i]=1./init_df[i,j]
                count+=1 
     
    for i in range(mat_size):
        for j in range(mat_size):
            a=getRealPosOfNode(my_order,i)
            b=getRealPosOfNode(my_order,j)
            if ((a==-1)or(b==-1)):
                my_df[i,j]=init_df[i,j]
            else:
                my_df[i,j]=init_df[a,b]
  
    # print("Dataframe:",init_df)    
    print("New Dataframe:",my_df) 
    return my_df

def sdmodLines2PwMatrix(my_model,raw_lines,start_ln,file_op=None,verbal=False):
    cur_id=[]
    line=start_ln
    matID=raw_lines[line]
    if(matID.isdigit()):
        my_cluster=my_model.clusters[int(matID)]
        cur_id=[my_cluster]
        if(verbal):
            print("reading cluster with ID"+str(matID))
    else:
        
        info=reqLib.re.split(r'[,\s]+', matID)
        int_conn_list = [int(elem) for elem in info if elem.strip()]
        par_cluster=my_model.clusters[int_conn_list[0]]
        my_node=my_model.clusters[int_conn_list[0]].nodes[int_conn_list[1]]
        con_cluster=my_model.clusters[int_conn_list[2]]
        
    line+= 1
    #tbd the ids in sdmod are not overall are with respect to the node
    component=raw_lines[line]
    info=reqLib.re.split(r'[,\s]+', component)
    int_cl_list = [int(elem) for elem in info if elem.strip()]


    clMatSize=int_cl_list[0]
    if(verbal): print(clMatSize)
    if(clMatSize>0 and matID.isdigit()==False):
        cur_id=[par_cluster,my_node,con_cluster]
        if(verbal):
                print(f"Node:{my_node.name} of cluster {par_cluster.name} is connected to cluster {con_cluster.name}")

    del int_cl_list[0]
    #order of clusters in matrix
    cl_labels=[]
    if (clMatSize>0):
        for i,item in enumerate(int_cl_list):
            if((i%2)==0):
                if(matID.isdigit()):
                    cur=my_model.clusters[item]
                else:
                    # print(con_cluster)
                    # print(item)
                    
                    cur=con_cluster.nodes[item]
                    #add connection my_node to cur
                    my_model.addNodeConnectionFromTo(my_node.name,cur.name,False)
                    if(file_op!=None):
                        file_op.write('\nnew_model.addNodeConnectionFromTo("'+my_node.name+'","'+cur.name+'",False)')
            else:
                cl_labels.insert(item,cur.name)
    # print(cl_labels)
    #this will give me the connection...check if nodes or clusters if cluster the cluster id but if node clA,node of A, clB
    if (clMatSize>1):
        cl_j=[]
        for item in range(clMatSize):
            line+=1
            component=raw_lines[line]
            info=reqLib.re.split(r'[,\s]+', component)
            cl_j_list = [float(elem) for elem in info if elem.strip()]
            # print(cl_j_list)
            cl_j+=cl_j_list
        line+=1
        pc_j_matrix =convertJudgVec2Pairwise(cl_j,int_cl_list,clMatSize)
        # cl_df = reqLib.pd.DataFrame(pc_j_matrix,cl_labels,cl_labels)
        cl_df = reqLib.pd.DataFrame(pc_j_matrix)
        if(verbal):
            print(cl_df)
    else:
        cl_df = reqLib.pd.DataFrame([])
        # cl_df=None
        line+=2
    return cl_df,cur_id,line

def reorder_pwc_np(model,np_array,list_conn):
    cnt=0
    new_array = []
    for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')):
            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                    
                    # print(f"Expecting CL A:{cluster.name} - ND: {nodeFrom.name} - CL B: {clusterPWC.name}")
                    # print(f"I have CL A:{list_conn[cnt][0].name} - ND: {list_conn[cnt][1].name} - CL B: {list_conn[cnt][2].name}") 
                    pos=find_index_pwc(list_conn,cluster,nodeFrom,clusterPWC)
                    # print(pos)
                    new_array.insert(cnt,np_array[pos])
                    # print(new_array)
                    cnt+=1
    return new_array

def find_index_pwc(lst, cluster_a, node_a, cluster_b):
    for i, sublst in enumerate(lst):
        if sublst == [cluster_a, node_a, cluster_b]:
            # print(f"I found {cluster_a} -{node_a} -{cluster_b} in {i}\n")
            return i
    return -1

def convIndex2ExcelRef(row,col):
    col_letter = reqLib.get_column_letter(col)
    excel_ref=col_letter+str(row)
    # print(f"c={col}-->letter{col_letter}")
    return excel_ref

def convExcelRef2Index(excel_ref):
    col = ''.join([i for i in excel_ref if not i.isdigit()])
    row = int(''.join([i for i in excel_ref if i.isdigit()]))
    return row,col

def set_cell_value_and_style(cell, value, fill=None, border=None, font_color=None, font_type=False):
    cell.value = value
    if fill:
        cell.fill = fill
    if border:
        cell.border = border
    if font_color:
        cell.font=reqLib.Font(color=font_color, bold=font_type)
        
def addDropDown2ExcelCell(worksheet,c_from,c_to,worksheet_title,cell_to_validate): 

   # Set up the data validation for the desired cell
   dv = reqLib.DataValidation(type="list", formula1=f'={worksheet_title}!{c_from}:{c_to}')
   worksheet.add_data_validation(dv)
   dv.add(worksheet[cell_to_validate])

   return worksheet

