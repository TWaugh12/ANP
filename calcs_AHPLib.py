from AhpAnpLib import functions_AHPLib as reqLib
from AhpAnpLib import inputs_AHPLib as cdf_inp
from AhpAnpLib import structs_AHPLib as cdf_str
# import functions_AHPLib as reqLib
# import inputs_AHPLib as cdf_inp
# import structs_AHPLib as cdf_str
#based on Bill Adams initial pyanp calculations

def vector2matrix(vector,size):
    cnt=0

    matrix = reqLib.np.ones((size, size))
    for i in range(0,size):
        for j in range (i+1,size):
            # print(vector[cnt])
            matrix[i,j]=vector[cnt]
            if(float(vector[cnt])>0):
                matrix[j,i]=1./vector[cnt]
            else:
                matrix[j,i]=0
            cnt+=1
    # print(matrix)
    return matrix
def matrixRaise2Power(matrixIn,power,rescale=False):
    last = reqLib.cp.deepcopy(matrixIn)
    nextm = reqLib.cp.deepcopy(matrixIn)
    count=1
    while count < power:
        reqLib.np.matmul(last, last, nextm)
        if rescale:
            mmax=reqLib.np.max(nextm)
            if mmax!=0:
                factor=1/mmax
                reqLib.np.multiply(nextm,factor,nextm)
        tmp = last
        last = nextm
        nextm = tmp
        count *= 2
    return last

def normalize(matrixIn):
    div = matrixIn.sum(axis=0)
    # print(f"In: {matrixIn} Div: {div}")
    for i in range(len(div)):
        if div[i] == 0:
            div[i] = 1.0
    matrixOut=matrixIn/div
    return matrixOut

def calcStart(matrixIn):
    #By Bill Adams
    n=len(matrixIn)
    if n<=0:
        # no entries...so go with 1
        return 1
    epsilon=1/(20*n)
    small_entries=[]
    for row in matrixIn:
        for val in row:
            if(reqLib.np.abs(val) <epsilon) and(val!=0):
                small_entries.append(reqLib.np.abs(val))
    if len(small_entries) <=0:
        # no entries
        return 20*n*n+10
    avg_smalls=reqLib.np.mean(small_entries)
    A=1/avg_smalls
    start_power=int(A) *n*n
    return start_power

def columnDistance(matrix1, matrix2, temp1=None,temp2=None,temp3=None):
    #column normalize matrix1 and 2 and calculate the distance, temps hold the normalized version of the vectors
    temp1=temp1 if temp1 is not None else reqLib.cp.deepcopy(matrix1)
    temp2=temp2 if temp2 is not None else reqLib.cp.deepcopy(matrix1)
    temp3=temp3 if temp3 is not None else reqLib.cp.deepcopy(matrix1)
    div1=matrix1.max(axis=0)
    div2=matrix2.max(axis=0)
    for i in range(len(div1)):
        if div1[i]==0:
            div1[i]=1
        if div2[i] ==0:
            div2[i]=1
    reqLib.np.divide(matrix1, div1, temp1)
    reqLib.np.divide(matrix2, div2, temp2)
    reqLib.np.subtract(temp1, temp2, temp3)
    reqLib.np.absolute(temp3, temp3)
    return reqLib.np.max(temp3)

def calcLimitANP(matrixIn,model=None,error=1e-10, max_iters=5000):
    
    #matrixIn scaled supermatrix
    size=len(matrixIn)
    difference=0.0
    start_power=calcStart(matrixIn)
    start=matrixRaise2Power(matrixIn,start_power,rescale=True)
    print("start pow",start_power)
    raised_big_power=matrixRaise2Power(matrixIn,size)

    #not a hierarchy
    if reqLib.np.count_nonzero(raised_big_power)==0:
        print("Matrix is a  Hierarchy ")
        return calcHierarchy(matrixIn)
   
    tmp1=reqLib.cp.deepcopy(matrixIn)
    tmp2=reqLib.cp.deepcopy(matrixIn)
    tmp3=reqLib.cp.deepcopy(matrixIn)

    pows=[start]
    for i in range(size-1):
        print(reqLib.np.matmul(matrixIn, pows[-1]))
        pows.append(reqLib.np.matmul(matrixIn, pows[-1]))
        difference=columnDistance(pows[-1], pows[-2], tmp1, tmp2, tmp3)
        if difference<error:
            # Already converged, done
            mysum=pows[-1].sum(axis=0)
            for i in range(len(mysum)):
                if mysum[i]==0:
                    mysum[i]=1
            if model is not None:
                    model.limitmatrix=reqLib.cp.deepcopy(pows[-1] /mysum)
            return pows[-1] /mysum
    for count in range(max_iters):
        nextp=pows[0]
        reqLib.np.matmul(pows[-1], matrixIn, nextp)
        for i in range(len(pows) -1):
            pows[i] =pows[i+1]
        pows[-1] =nextp
        # Check convergence
        for i in range(len(pows) -1):
            difference=columnDistance(pows[i], nextp, tmp1, tmp2, tmp3)
            if difference<error:
                mysum=nextp.sum(axis=0)
                for i in range(len(mysum)):
                    if mysum[i] ==0:
                        mysum[i] =1
                print("Count was "+str(count))
                if model is not None:
                    model.limitmatrix=reqLib.cp.deepcopy(nextp/mysum)
                return nextp/mysum    
    raise ValueError("Did not converge")

def calcHierarchy(matrixIn,model=None):
    size=len(matrixIn)
    raised_big_power=matrixRaise2Power(matrixIn,size)
    #initialize
    total=reqLib.cp.deepcopy(matrixIn)
    this_power=reqLib.cp.deepcopy(matrixIn)
    next_power=reqLib.cp.deepcopy(matrixIn)
    #not a hierarchy
    if reqLib.np.count_nonzero(raised_big_power)!=0:
        print("Matrix not a  Hierarchy ")
        diff = 1
        error:float = 1e-10
        vec = reqLib.np.ones([size])
        while diff > error:
            reqLib.np.matmul(this_power,matrixIn,next_power)
            temp=this_power
            this_power=next_power
            next_power=temp
            
            nextv = reqLib.np.matmul(matrixIn, vec)
            nextv = nextv/sum(nextv)
            diff = max(abs(nextv - vec))
            vec = nextv
        anp_result=vec
        limit_result=normalize(this_power)
        print("Anp result",anp_result)
        print("Limit matrix",limit_result)
        if model is not None:
            model.limitmatrix=reqLib.cp.deepcopy(limit_result)
        return limit_result
    
    #raise to powers and sum up
    for i in range(size-2):
        reqLib.np.matmul(this_power,matrixIn,next_power)
        reqLib.np.add(total,next_power,total)
        temp=this_power
        this_power=next_power
        next_power=temp
    result=normalize(total)
    if model is not None:
        model.limitmatrix=reqLib.cp.deepcopy(result)
    return result
def harkerFix(matrixIn):
    nrows = matrixIn.shape[0]
    ncols = matrixIn.shape[1]
    fixed = matrixIn.copy()
    for row in range(nrows):
        val = 1
        for col in range(ncols):
            if col != row and matrixIn[row,col]==0:
                val+=1
        fixed[row,row]=val
    return(fixed)
def priorityVector(matrixIn,harker=True,error:float = 1e-10):
    if matrixIn is None or matrixIn.shape==(0,0):
        # Eigen vector of the empty matrix is []
        return reqLib.np.array([])
    if harker:
        matrixIn = harkerFix(matrixIn)
    size = matrixIn.shape[0]

    #Create our return value
    vec = reqLib.np.ones([size])
    diff = 1
    while diff > error:
        nextv = reqLib.np.matmul(matrixIn, vec)
        nextv = nextv/sum(nextv)
        diff = max(abs(nextv - vec))
        vec = nextv
   
    return(vec)

def nodeNameList(model):
    nodesAll=model.getNodeListFromClusters()
    nodeNum=len(nodesAll)
    nodeNames=[]
    for node in nodesAll:
        nodeNames.append(node.name)
    return nodeNames
def clusterNameList(model):
    #not unique cluster names -one time per node in the cluster
    clustersAll=model.getClusterListFromClusters()
    
    clusterNames=[]
    for cluster in clustersAll:
        clusterNames.append(cluster.name)
    return clusterNames
def uniqueClusterNameList(model):
    #unique cluster names
    clustersAll=model.getClusterListFromClusters()
    
    clusterNames=[]
    for cluster in clustersAll:
        if(len(clusterNames)>0):
            if(cluster.name!=clusterNames[-1]):
                clusterNames.append(cluster.name)
        else:
            clusterNames.append(cluster.name)
    return clusterNames
def pwcMatrixCol(model,nodeCol,nodeRow):
    pcMat=0
    for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')):
        for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
            connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
            for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                pcMat=0
                for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                    
                    if nodeFrom.nodeID==nodeCol.nodeID and nodeA.nodeID==nodeRow.nodeID:
                        return pcMat
                    if(nodeA in nodeFrom.connectedTo):
                        pcMat+=1
    return -1

def pwcMatrixRow(model,nodeCol,nodeRow):
    pcMatR=0
    for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')):
        for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
            connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
            for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
               
                for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                    
                    if nodeFrom.nodeID==nodeCol.nodeID and nodeA.nodeID==nodeRow.nodeID:
                        return pcMatR
                pcMatR+=1

    return -1                   

def calcUnweightedSuperMatrix(model, verbal=False):
    nodesAll=model.getNodeListFromClusters()
    nodeNum=len(nodesAll)
    super=reqLib.np.zeros([nodeNum,nodeNum])
    for matrixIn in model.all_pc_matrices:
        eigen=priorityVector(matrixIn)
        model.all_pr_vectors.append(eigen)
    
    col=0

    

    for nodeCol in nodesAll:
        row=0

        for nodeRow in nodesAll:
            if nodeRow in nodeCol.connectedTo:
                pwcR=pwcMatrixRow(model,nodeCol,nodeRow)
                pwcC=pwcMatrixCol(model,nodeCol,nodeRow)
                super[row][col]= model.all_pr_vectors[pwcR][pwcC]
            row+=1
        col+=1
    if(verbal==True):
        print(super)
    model.supermatrix=reqLib.cp.deepcopy(super)
    return super
def calcWeightedSupermatrix(model):
    weight=normalize(model.supermatrix)
    model.weighted_supermatrix=reqLib.cp.deepcopy(weight)
    return weight
def calcLimitingPriorities(supermatrixIn,verbal=False):
    size=len(supermatrixIn)
    raised_big_power=matrixRaise2Power(supermatrixIn,size)
    limiting=[]
    #not a hierarchy
    # if reqLib.np.count_nonzero(raised_big_power)!=0:
    #     print("Matrix not a  Hierarchy ")
    #     #calc limit matrix
    # else:
    result=calcHierarchy(supermatrixIn)
    limiting=result[0:size,0]
    if (verbal==True):
        print(limiting)
    return limiting

def calcPrioritiesNormalizedByCluster(supermatrixIn,model):
    size=len(supermatrixIn)
    raised_big_power=matrixRaise2Power(supermatrixIn,size)
    limiting=[]
    nodesAll=model.getNodeListFromClusters()
    nodeNum=len(nodesAll)
    byCluster=reqLib.np.zeros(nodeNum)
    #not a hierarchy
    # if reqLib.np.count_nonzero(raised_big_power)!=0:
    #     print("Matrix not a  Hierarchy ")
    #     #calc limit matrix
    # else:
    result=calcHierarchy(supermatrixIn)
    limiting=result[0:size,0]
    
    nodesbefore=0
    
    for cluster in sorted(model.clusters, key=reqLib.op.attrgetter('order')):
        
        i=0
        cl=0
        sum=0.000
        for node in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')):
            sum+=limiting[i+nodesbefore]
            # print("lim",limiting[i+nodesbefore],"--",i+nodesbefore)
            i+=1
        # print("sum",sum)
        i=0
        for node in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')):
            if sum!=0:
                # print("lim",limiting[i+nodesbefore])
                limiting[i+nodesbefore]=limiting[i+nodesbefore]/sum
                byCluster[i+nodesbefore]=limiting[i+nodesbefore]
    
                # print("lim",limiting[i+nodesbefore])
            i+=1
        nodesbefore+=len(cluster.nodes)
    # print(byCluster)  
      
    return byCluster  

def calcPrioritiesOfCluster(supermatrixIn,clusterName,model, verbal=False):
    size=len(supermatrixIn)
    raised_big_power=matrixRaise2Power(supermatrixIn,size)
    limiting=[]
    synth=reqLib.np.ones(1)
    #not a hierarchy
    # if reqLib.np.count_nonzero(raised_big_power)!=0:
    # #     print("Matrix not a  Hierarchy ")
    # #     #calc limit matrix
    # # else:
    result=calcHierarchy(supermatrixIn)
    limiting=result[0:size,0]
    start=0
    for cluster in sorted(model.clusters, key=reqLib.op.attrgetter('order')):
        if clusterName==cluster.name:
            sum=0.0
            i=0
            synth=reqLib.np.zeros(len(cluster.nodes))
            for node in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')):
                sum+=limiting[start+i]
                # print(sum)
                i+=1
            row=0
            for node in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')):
                if sum!=0.0:
                    # print("!",limiting[row+start])
                    synth[row]=limiting[start+row]/sum
                    # print("@",synth[row])
                    row+=1
        start+=len(cluster.nodes)
    if(verbal==True):
        print("Synthesized results:",synth)
    return synth

def calcSumOfCols(matrixIn):
    size=len(matrixIn)
    sum_v= reqLib.np.zeros(size)
    for j in range (0,size):
        for i in range (0, size):
            sum_v[j]+=matrixIn[i,j]
    # print(sum_v)
    return sum_v

def calcInconsistency(matrixIn,verbal=False):
    size=len(matrixIn)
    priority_v=priorityVector(matrixIn)
    col_sum_v=calcSumOfCols(matrixIn)
    lmax=col_sum_v@priority_v
    ci=(lmax-size)/(size-1)
    if(size<15):
        inconsistency=ci/RI(size)
    else:
        inconsistency=ci/RI(15)

    if(verbal==True):
        print(matrixIn)
        # print(RI(size))
        # print(priority_v)
        # print(col_sum_v)
        print("lmax=",round(lmax,3))
        print("CI= ",round(ci,3))
        print("Inconsistency= ",round(inconsistency,3))
    return(round(inconsistency,3))
def RI(size):
    RI_matrix=[1,1,1,0.52,0.89,1.12,1.25,1.35,1.4,1.45,1.49,1.51,1.54,1.56,1.57,1.58]
    return RI_matrix[size]

def copyExcelSheet(FromWorkbookPath, ToWorkbookPath, sheetName, cpRes=True):

    #styles
    border = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    # Open the existing workbooks

    if reqLib.os.path.isfile(FromWorkbookPath):
        source_wb = reqLib.openpyxl.load_workbook(FromWorkbookPath)
        source_ws = source_wb.active
         
        if reqLib.os.path.isfile(ToWorkbookPath):
            # print("load")
            destination_wb = reqLib.openpyxl.load_workbook(ToWorkbookPath)
            destination_ws = destination_wb.active
            destination_ws.title = sheetName
        else:
            # print("creating")
            destination_wb = reqLib.openpyxl.Workbook()
            destination_ws = destination_wb.active
            destination_ws.title = sheetName
        if(cpRes==False):
            max_col = source_ws.max_column-4
            #-4 if you don't want to show the results
        else:
            max_col = source_ws.max_column
        # Copy the cells from the source worksheet to the destination worksheet
        for row in source_ws.iter_rows(max_col=max_col):
            for cell in row:
                # print("....")
                destination_ws[cell.coordinate].value = cell.value
                if cell.has_style:
                    destination_ws[cell.coordinate].font = reqLib.copy(cell.font)
                    destination_ws[cell.coordinate].border = reqLib.copy(cell.border)
                    destination_ws[cell.coordinate].fill = reqLib.copy(cell.fill)
                    destination_ws[cell.coordinate].number_format = reqLib.copy(cell.number_format)
                    if(cell.value is not None):
                        if(type(cell.value) in [float,int]):
                            destination_ws[cell.coordinate].border = border
        destination_ws.column_dimensions['A'].width = 15.00
        destination_wb.save(ToWorkbookPath)

    # Save the new workbook
    else:
        print("Source File not found")
 

def calcSensOfAll(model):
    inc_index=[]
    for mat in model.all_pc_matrices:
        inc_index.append(calcInconsistency(mat))
    return inc_index
def calcSensOfAllScales(model):
    inc_index=[]
    for crit in model.rateModel.ratCriteria:
        scale=model.rateModel.getScaleOfCriterion(crit)
        print(scale.name,scale.val_mat)
        inc_index.append(calcInconsistency(scale.val_mat))
        print(inc_index)
    return inc_index

def calcPriorNormalVectOfAll(model):
    pwc_eigen=[]
    for mat in model.all_pc_matrices:
        pwc_eigen.append(priorityVector(mat))
    return pwc_eigen

def calcPriorNormalVectOfAllScales(model):
    sc_eigen=[]
    if(model.modelType=="ratings"):
            for my_crit in model.rateModel.ratCriteria:
                my_scale=model.rateModel.getScaleOfCriterion(my_crit)
                sc_eigen.append(priorityVector(my_scale.val_mat))
    print(sc_eigen)
    return sc_eigen

def calcPriorIdealVectOfAll(model):
    pwc_eigen=[]
    for mat in model.all_pc_matrices:
        pwc_eigen.append(idealVector(priorityVector(mat)))
    return pwc_eigen
def calcPriorIdealVectOfAllScales(model):
    sc_eigen=[]
    if(model.modelType=="ratings"):
            for my_crit in model.rateModel.ratCriteria:
                my_scale=model.rateModel.getScaleOfCriterion(my_crit)
                sc_eigen.append(idealVector(priorityVector(my_scale.val_mat)))
    print(sc_eigen)
    return sc_eigen
def idealVector(vector):
    max=0
    ideal=[]
    for v in vector:
         if(v>max):
            max=v
    if (max!=0):
        for v in vector:
             ideal.append(v/max)
    return ideal
        


def appendResults2Excel(model,filepath,worksheetName, normalbar=False,idealbar=True,verb=False):

    v_incons_all=calcSensOfAll(model)
    v_eigen_all=calcPriorNormalVectOfAll(model)
    v_eigen_ideal_all=calcPriorIdealVectOfAll(model)

    border = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    border_noright = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style=None), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    border_noleft = reqLib.Border(left=reqLib.Side(style=None), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    no_border = reqLib.Border(left=reqLib.Side(border_style=None), 
                   right=reqLib.Side(border_style=None), 
                   top=reqLib.Side(border_style=None), 
                   bottom=reqLib.Side(border_style=None))

    res_hd_fill = reqLib.PatternFill(start_color="ffb6e5f2", end_color="ffb6e5f2", fill_type="solid")
    res_fill = reqLib.PatternFill(start_color="ffd1d5de", end_color="ffd1d5de", fill_type="solid")
    inc_fill = reqLib.PatternFill(start_color="ffffe4f7", end_color="ffffe4f7", fill_type="solid")
    empty_fill = reqLib.PatternFill(start_color="FFFFFFFF", end_color="FFFFFFFF", fill_type="solid")
    rule_n = reqLib.DataBarRule(start_type='min', start_value=0, end_type='max', end_value=0,
                   color=reqLib.Color(rgb='fff2463a'), showValue=True)
    rule_i = reqLib.DataBarRule(start_type='min', start_value=0, end_type='max', end_value=0,
                   color=reqLib.Color(rgb='ff0b30b5'), showValue=True)
    prw_matrix_cnt=0
    if reqLib.os.path.isfile(filepath):
        # print("load")
        destination_wb = reqLib.openpyxl.load_workbook(filepath)
        destination_ws = destination_wb.active
        destination_ws.title = worksheetName
        # worksheet.set_column(0,20,15)

        row=0
        col=1
        for cluster in sorted( model.clusters, key=reqLib.op.attrgetter('order')):
            for nodeFrom in sorted(cluster.nodes, key=reqLib.op.attrgetter('order')): 
                connectedClusters=model.retAllClusterConnectionsFromNode(nodeFrom.name)
                if(len(connectedClusters)>0):
                    row=row+1
                    for clusterPWC in sorted(connectedClusters, key=reqLib.op.attrgetter('order')):
                        # print("Nodes:\n",clusterPWC.nodes)
                        row=row+1
                        col=0
                        size=0
                        for nodeA in sorted(clusterPWC.nodes, key=reqLib.op.attrgetter('order')):
                            # print("NodeA: ",nodeA.nodeID)         
                            if(nodeA in nodeFrom.connectedTo):
                                col+=1
                                size+=1
                        row_res=row
                        col_res=size+5
                        for a in range(size):
                                for b in range (size):
                                    destination_ws.cell(row=row_res+a+2, column=b+2).value =round(model.all_pc_matrices[prw_matrix_cnt][a,b],3)
                        destination_ws.cell(row=row_res, column=1).value = None
                        destination_ws.cell(row=row_res, column=col_res).value = 'Results'
                        destination_ws.cell(row=row_res, column=col_res).border = border_noright
                        destination_ws.cell(row=row_res, column=col_res).fill = res_hd_fill
                        
                        destination_ws.cell(row=row_res, column=col_res+1).border = border_noleft
                        destination_ws.cell(row=row_res, column=col_res+1).fill = res_hd_fill

                        row_hd=row+1
                        #clean up estimates
                        destination_ws.cell(row=row_hd, column=col_res-1).value = None
                        destination_ws.cell(row=row_hd, column=col_res-2).value = None
                        destination_ws.cell(row=row_hd, column=col_res-1).fill = empty_fill
                        destination_ws.cell(row=row_hd, column=col_res-2).fill = empty_fill
                        destination_ws.cell(row=row_hd, column=col_res-1).border = no_border
                        destination_ws.cell(row=row_hd, column=col_res-2).border = no_border
                        #actual result headers
                        destination_ws.cell(row=row_hd, column=col_res).value = 'Normal'
                        destination_ws.cell(row=row_hd, column=col_res).border = border
                        destination_ws.cell(row=row_hd, column=col_res).fill = res_fill
                        
                        destination_ws.cell(row=row_hd, column=col_res+1).value = 'Ideal'
                        destination_ws.cell(row=row_hd, column=col_res+1).border = border
                        destination_ws.cell(row=row_hd, column=col_res+1).fill = res_fill
                        
                        
                        for cnt in range(1,size+1):
                            destination_ws.cell(row=row_hd+cnt, column=col_res-1).value = None
                            destination_ws.cell(row=row_hd+cnt, column=col_res-1).fill = empty_fill
                            destination_ws.cell(row=row_hd+cnt, column=col_res-2).value = None
                            destination_ws.cell(row=row_hd+cnt, column=col_res-2).fill = empty_fill
                            destination_ws.cell(row=row_hd+cnt, column=col_res-1).border = no_border
                            destination_ws.cell(row=row_hd+cnt, column=col_res-2).border = no_border
                            
                            destination_ws.cell(row=row_hd+cnt, column=col_res).value = round(v_eigen_all[prw_matrix_cnt][cnt-1],3)
                            destination_ws.cell(row=row_hd+cnt, column=col_res).border = border
                            destination_ws.cell(row=row_hd+cnt, column=col_res).fill = res_fill
                            destination_ws.cell(row=row_hd+cnt, column=col_res+1).value = round(v_eigen_ideal_all[prw_matrix_cnt][cnt-1],3)
                            destination_ws.cell(row=row_hd+cnt, column=col_res+1).border = border
                            destination_ws.cell(row=row_hd+cnt, column=col_res+1).fill = res_fill
                        #clear est inc and sums
                        for cnt in range(size+1,size+3):
                            destination_ws.cell(row=row_hd+cnt, column=col_res-1).value = None
                            destination_ws.cell(row=row_hd+cnt, column=col_res-1).fill = empty_fill
                            destination_ws.cell(row=row_hd+cnt, column=col_res-1).border = no_border
                            destination_ws.cell(row=row_hd+cnt, column=col_res-2).value = None
                            destination_ws.cell(row=row_hd+cnt, column=col_res-2).fill = empty_fill
                            destination_ws.cell(row=row_hd+cnt, column=col_res-2).border = no_border
                        # print(normalbar,idealbar)
                        if(normalbar==True):
                            cell_ref_norm=destination_ws.cell(row=row_hd+1, column=col_res).coordinate+":"+destination_ws.cell(row=row_hd+size, column=col_res).coordinate
                            destination_ws.conditional_formatting.add(cell_ref_norm,rule_n)
                        if(idealbar==True):
                            cell_ref_ideal=destination_ws.cell(row=row_hd+1, column=col_res+1).coordinate+":"+destination_ws.cell(row=row_hd+size, column=col_res+1).coordinate
                            destination_ws.conditional_formatting.add(cell_ref_ideal,rule_i)
                            
                        #destination_ws.conditional_formatting.add( destination_ws.cell(row=row_hd+1, column=col_res), destination_ws.cell(row=row_hd+size+1, column=col_res), rule)

                        row_inc=row+size+2
                        col_inc=size+5
                        destination_ws.cell(row=row_inc, column=col_inc).value = 'Incons.'
                        destination_ws.cell(row=row_inc, column=col_inc).border = border_noright
                        destination_ws.cell(row=row_inc, column=col_inc).fill = inc_fill

                        destination_ws.cell(row=row_inc, column=col_inc+1).value = round(v_incons_all[prw_matrix_cnt],3)
                        destination_ws.cell(row=row_inc, column=col_inc+1).border = border_noleft
                        destination_ws.cell(row=row_inc, column=col_inc+1).fill = inc_fill

                        
                        prw_matrix_cnt+=1
                        row=row+size+3
                    if verb:
                        print("Saved inconsistency for cluster-node : ",clusterPWC.name, "-", nodeFrom.name) 
                # row+=1
        destination_wb.save(filepath)
        if(verb):
            print("---------------------------------------------------------------------------\n")
            print("Inconsistency indices:")
            print(v_incons_all)
            print("---------------------------------------------------------------------------\n")
            print("Priority vectors:")
            print(v_eigen_all)

def appendScaleResults2Excel(model,filepath,worksheetName, normalbar=False,idealbar=True,verb=False):

    v_incons_all=calcSensOfAllScales(model)
    v_eigen_all=calcPriorNormalVectOfAllScales(model)
    v_eigen_ideal_all=calcPriorIdealVectOfAllScales(model)

    border = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    border_noright = reqLib.Border(left=reqLib.Side(style='thin'), right=reqLib.Side(style=None), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    border_noleft = reqLib.Border(left=reqLib.Side(style=None), right=reqLib.Side(style='thin'), top=reqLib.Side(style='thin'), bottom=reqLib.Side(style='thin'))
    res_hd_fill = reqLib.PatternFill(start_color="ffb6e5f2", end_color="ffb6e5f2", fill_type="solid")
    res_fill = reqLib.PatternFill(start_color="ffd1d5de", end_color="ffd1d5de", fill_type="solid")
    inc_fill = reqLib.PatternFill(start_color="ffffe4f7", end_color="ffffe4f7", fill_type="solid")
    
    rule_n = reqLib.DataBarRule(start_type='min', start_value=0, end_type='max', end_value=0,
                   color=reqLib.Color(rgb='fff2463a'), showValue=True)
    rule_i = reqLib.DataBarRule(start_type='min', start_value=0, end_type='max', end_value=0,
                   color=reqLib.Color(rgb='ff0b30b5'), showValue=True)
    matrix_cnt=0
    
    if reqLib.os.path.isfile(filepath):
        # print("load")
        destination_wb = reqLib.openpyxl.load_workbook(filepath)
        destination_ws = destination_wb.active
        destination_ws.title = worksheetName
        # worksheet.set_column(0,20,15)
        row=1
        col=1
        if(model.modelType=="ratings"):
            for my_crit in model.rateModel.ratCriteria:
                my_scale=model.rateModel.getScaleOfCriterion(my_crit)
                if(verb):
                    print(f"Criterion: {my_crit}")
                    print(my_scale)
                size=len(my_scale.members)
               
                #results headers
                row_res=row
                col_res=size+4
                destination_ws.cell(row=row_res, column=col_res).value = 'Results'
                destination_ws.cell(row=row_res, column=col_res).border = border_noright
                destination_ws.cell(row=row_res, column=col_res).fill = res_hd_fill
                
                destination_ws.cell(row=row_res, column=col_res+1).border = border_noleft
                destination_ws.cell(row=row_res, column=col_res+1).fill = res_hd_fill

                row_hd=row+1
                destination_ws.cell(row=row_hd, column=col_res).value = 'Normal'
                destination_ws.cell(row=row_hd, column=col_res).border = border
                destination_ws.cell(row=row_hd, column=col_res).fill = res_fill
                
                destination_ws.cell(row=row_hd, column=col_res+1).value = 'Ideal'
                destination_ws.cell(row=row_hd, column=col_res+1).border = border
                destination_ws.cell(row=row_hd, column=col_res+1).fill = res_fill
                
                #inconsistency
                row_inc=row+size+2
                col_inc=size+4
                destination_ws.cell(row=row_inc, column=col_inc).value = 'Incons.'
                destination_ws.cell(row=row_inc, column=col_inc).border = border_noright
                destination_ws.cell(row=row_inc, column=col_inc).fill = inc_fill

                destination_ws.cell(row=row_inc, column=col_inc+1).value = round(v_incons_all[matrix_cnt],3)
                destination_ws.cell(row=row_inc, column=col_inc+1).border = border_noleft
                destination_ws.cell(row=row_inc, column=col_inc+1).fill = inc_fill
                
                if verb:
                    print("Saved inconsistency for cluster-node : ",my_crit.name, "-", my_scale.name) 
            
                
                for cnt in range(1,size+1):
                    destination_ws.cell(row=row_hd+cnt, column=col_res).value = round(v_eigen_all[matrix_cnt][cnt-1],3)
                    destination_ws.cell(row=row_hd+cnt, column=col_res).border = border
                    destination_ws.cell(row=row_hd+cnt, column=col_res).fill = res_fill
                    destination_ws.cell(row=row_hd+cnt, column=col_res+1).value = round(v_eigen_ideal_all[matrix_cnt][cnt-1],3)
                    destination_ws.cell(row=row_hd+cnt, column=col_res+1).border = border
                    destination_ws.cell(row=row_hd+cnt, column=col_res+1).fill = res_fill
                    # print(normalbar,idealbar)
                    if(normalbar==True):
                        cell_ref_norm=destination_ws.cell(row=row_hd+1, column=col_res).coordinate+":"+destination_ws.cell(row=row_hd+size, column=col_res).coordinate
                        destination_ws.conditional_formatting.add(cell_ref_norm,rule_n)
                    if(idealbar==True):
                        cell_ref_ideal=destination_ws.cell(row=row_hd+1, column=col_res+1).coordinate+":"+destination_ws.cell(row=row_hd+size, column=col_res+1).coordinate
                        destination_ws.conditional_formatting.add(cell_ref_ideal,rule_i)
                row=row+size+5
                matrix_cnt+=1
            destination_wb.save(filepath)
            if(verb):
                print("---------------------------------------------------------------------------\n")
                print("Inconsistency indices:")
                print(v_incons_all)
                print("---------------------------------------------------------------------------\n")
                print("Normal Priority vectors:")
                print(v_eigen_all)
                print("---------------------------------------------------------------------------\n")
                print("Ideal Priority vectors:")
                print(v_eigen_ideal_all)
        else:
            print("Cannot calculate ratings for this model. You need to define the model as a ratings model")
    else:
        print("File not found/not able to access while trying to export rating scales results to existing file:",filepath)
        
def appendRateTblResults2Excel(model,filepath,worksheetName, normalbar=False,idealbar=True,verb=False):

    
    if reqLib.os.path.isfile(filepath):
        # print("load")
        destination_wb = reqLib.openpyxl.load_workbook(filepath)
        destination_ws = destination_wb.active
        destination_ws.title = worksheetName
        # worksheet.set_column(0,20,15)
        scale_indx_tbl=model.rateModel.ratMatrix
        row=0
        col=0
        if(model.modelType=="ratings"):
            for i,alt in enumerate(model.rateModel.ratAlternatives):       
                for j,crit in enumerate(model.rateModel.ratCriteria):
                    
                    scale=model.rateModel.getScaleOfCriterion(crit)
                    destination_ws.cell(i+2, j+2).value = scale.members[scale_indx_tbl[i,j]][0]
            destination_wb.save(filepath)
            if(verb):
                print("---------------------------------------------------------------------------\n")
                print("Rating scale indices:")
                print(scale_indx_tbl)
                print("---------------------------------------------------------------------------\n")
                print("Rating Table:")
                # print(rate_tbl)


        else:
            print("Cannot calculate ratings table for this model. You need to define the model as a ratings model")
    else:
        print("File not found/not able to access while trying to export rating scales results to existing file:",filepath)
  
def updateRateTblResults2Excel(model,filepath,worksheetName, verb=False):

    
    if reqLib.os.path.isfile(filepath):
        # print("load")
        destination_wb = reqLib.openpyxl.load_workbook(filepath)
        destination_ws = destination_wb.active
        destination_ws.title = worksheetName
        # worksheet.set_column(0,20,15)

        scale_indx_tbl=model.rateModel.ratMatrix
        if len(scale_indx_tbl)>0:
            row=0
            col=0
            if(model.modelType=="ratings"):
                for i,alt in enumerate(model.rateModel.ratAlternatives):       
                    for j,crit in enumerate(model.rateModel.ratCriteria):
                        
                        scale=model.rateModel.getScaleOfCriterion(crit)
                        if i < len(scale_indx_tbl) and j < len(scale_indx_tbl[0]):
                            destination_ws.cell(i+2, j+2).value = scale.members[scale_indx_tbl[i,j]][0]
                destination_wb.save(filepath)
                if(verb):
                    print("---------------------------------------------------------------------------\n")
                    print("Rating scale indices:")
                    print(scale_indx_tbl)
                    print("---------------------------------------------------------------------------\n")
                    print("Rating Table:")
                    # print(rate_tbl)


            else:
                print("!!!Cannot calculate ratings table for this model. You need to define the model as a ratings model")
        else:
            print("No ratings tables available to load")
    else:
        print("File not found/not able to access while trying to export rating table results to existing file:",filepath)
              
def calcAHPMatricesSave2File(model,inputFile,filepath,inputFileUse=True,normalbar=False,idealbar=True,verbal=False):
    if(inputFileUse):
        cdf_inp.importFromExcel(model,inputFile,0,verbal)
        #add the inputs to the results file
        
        copyExcelSheet(inputFile, filepath, "pairwise_comp")
    else:
      
        cdf_inp.export4ExcelQuestFull(model,filepath)

    

    
     #add inconsistency  to the results file
    # add priority vectors in the order given - assuming no changes

    appendResults2Excel(model,filepath,"pairwise_comp",normalbar,idealbar,verbal)
    
    model.drawGraphModel(filepath)
    listTitles=nodeNameList(model)
    clusterTitles=clusterNameList(model)
    nodesInClusters=model.getListNumNodesInClusters()

    float_format = "%.3f"
    # print(nodesInClusters)

    #calculate unweighted super matrix
    super=calcUnweightedSuperMatrix(model)
    
    df = reqLib.pd.DataFrame (super,index=listTitles,columns=listTitles)

    with reqLib.pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:  
        df.to_excel(writer, sheet_name="supermatrix",startrow=2 - 1, startcol=ord("B") - 65, float_format=float_format)
    
    cdf_inp.addLabels2Excel(clusterTitles,filepath,"supermatrix",3,1,0)
    cdf_inp.addLabels2Excel(clusterTitles,filepath,"supermatrix",1,3,1)
    #calculate AHP results: limit matrix
    hierarchy = calcHierarchy(model.supermatrix)
    df = reqLib.pd.DataFrame (hierarchy,index=listTitles,columns=listTitles)
    with reqLib.pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:  
        df.to_excel(writer, sheet_name="limit matrix",startrow=2 - 1, startcol=ord("B") - 65, float_format=float_format)
    
    cdf_inp.addLabels2Excel(clusterTitles,filepath,"limit matrix",3,1,0)
    cdf_inp.addLabels2Excel(clusterTitles,filepath,"limit matrix",1,3,1)

    #calculate global priorities
    limit = calcLimitingPriorities(model.supermatrix)
    #print out 
    df = reqLib.pd.DataFrame (limit,index=listTitles,columns=["Limiting Prior."])
    with reqLib.pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:  
        df.to_excel(writer, sheet_name="limitingPriorities", startcol=ord("B") - 65, float_format=float_format)

    cdf_inp.addLabels2Excel(clusterTitles,filepath,"limitingPriorities",2,1,0)
    #calculate local priorities - normalized by cluster
    bycluster = calcPrioritiesNormalizedByCluster(model.supermatrix, model)
    df = reqLib.pd.DataFrame (bycluster,index=listTitles,columns=["Local Priorities"])

    with reqLib.pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:  
        df.to_excel(writer, sheet_name="localPriorities", startcol=ord("B") - 65, float_format=float_format)

    cdf_inp.addLabels2Excel(clusterTitles,filepath,"localPriorities",2,1,0)

    

    

    if (verbal):
        print("-----------------------SUPERMATRIX----------------------------------------------------\n")
        print(super)
        print("-----------------------HIERARCHY----------------------------------------------------\n")
        print(hierarchy)
        print("----------------------------LIMITING-----------------------------------------------\n")
        print(limit)
        print("---------------------------BY CLUSTER------------------------------------------------\n")
        print(bycluster)
                
def calcMatrices(model,inputFile,inputFileUse=True,verbal=False):
    #results saved internally not in file

    if(inputFileUse):
        cdf_inp.importFromExcel(model,inputFile,0)
    
    listTitles=nodeNameList(model)

    #calculate unweighted super matrix
    super=calcUnweightedSuperMatrix(model)
  
    
    #calculate AHP results: limit matrix
    hierarchy = calcHierarchy(model.supermatrix)
       
    #calculate global priorities
    limit = calcLimitingPriorities(model.supermatrix)
 
       #calculate local priorities - normalized by cluster
    bycluster = calcPrioritiesNormalizedByCluster(model.supermatrix, model)

        
    if (verbal):
        print("-----------------------SUPERMATRIX----------------------------------------------------\n")
        print(super)
        print("-----------------------HIERARCHY----------------------------------------------------\n")
        print(hierarchy)
        print("----------------------------LIMITING-----------------------------------------------\n")
        print(limit)
        print("---------------------------BY CLUSTER------------------------------------------------\n")
        print(bycluster)
def calcRateCritV(model,verbal=True):
    if(model.modelType=="ratings"):
            
            #calculate unweighted super matrix
            super=calcUnweightedSuperMatrix(model)
            #calculate AHP results: limit matrix
            hierarchy = calcHierarchy(model.supermatrix)       
            #calculate global priorities
            limit = calcLimitingPriorities(model.supermatrix)
            
            listTitles=nodeNameList(model)
            pwc_eigen=reqLib.np.empty(len(model.rateModel.ratCriteria))

            for ind1,my_crit in enumerate(model.rateModel.ratCriteria):
                for ind2,name in enumerate(listTitles):
                    if (name==my_crit.name):
                            pwc_eigen[ind1]=limit[ind2]
                
               #calc ideal priority vector for model
            model.rateModel.ratCritPriorities=normalize(reqLib.np.array(pwc_eigen.copy()).reshape(-1, 1)).flatten().tolist()
            
            if (verbal):
                print("-----------------------PRIORITIES----------------------------------------------------\n")
                # print(limit)
                # print(listTitles)
                print("Re-normalized priorities based on the selected criteria for the rating model")
                for i,crit in enumerate(model.rateModel.ratCriteria):
                    print (crit.name,": ",model.rateModel.ratCritPriorities[i])

def sensitivityCellSupermatrix(model,nodeName,verbal=False):
    matrixIn=model.supermatrix
    node_names=nodeNameList(model)
    if len(matrixIn) == 0:
        matrixIn=calcUnweightedSuperMatrix(model)
    influence=reqLib.np.arange(0.0, 1.1, 0.1)
    
    for i in range(0,len(influence)):
        working_matrix=reqLib.cp.deepcopy(matrixIn)
        working_matrix[node_names.index(nodeName)][0]=influence[i]
        # print(working_matrix)
        # working_matrix=normalize(working_matrix)
        super=calcHierarchy(working_matrix)
        if(verbal==True):
            print(influence[i],"--",super[:,0])

def sensitivityCellSupermatrixShort(model,clusterName,nodeName,verbal=False):
    matrixIn=model.supermatrix
    if len(matrixIn) == 0:
        matrixIn=calcUnweightedSuperMatrix(model)
    influence=reqLib.np.arange(0.0, 1.1, 0.1)
    points=[]
    node_names=nodeNameList(model)
    for i in range(0,len(influence)):
        working_matrix=reqLib.cp.deepcopy(matrixIn)
        # print(node_names,nodeName)
        original_value=working_matrix[node_names.index(nodeName)][0]
        new_value=influence[i]
        # what should be the multiplier when 1 the original value...something very large? 
        multiplier=0.000001
        if(new_value!=1):
         multiplier=(1.-new_value)/(1.-original_value)
        if(original_value==1):
            multiplier=(1.-new_value)/0.000001
        if(verbal==True):
            print("multiplier",multiplier)
            print("supermatrix",working_matrix)
        for item_i in range(0,len(matrixIn)):
            working_matrix[item_i][0]=working_matrix[item_i][0]*multiplier
        working_matrix[node_names.index(nodeName)][0]=influence[i]
        # print(working_matrix)
        # working_matrix=normalize(working_matrix)
        super=calcHierarchy(working_matrix)
        # print(influence[i],"--",super[:,0])
        if(verbal==True):
            print(influence[i])
        points.append(calcPrioritiesOfCluster(super,clusterName,model))
        if(verbal==True):
            print (points)
    return points

def sensitivityCellSupermatrixPlot(model,clusterName,filepath="show",verbal=False,*args):
    matrixIn=model.supermatrix
    if len(matrixIn) == 0:
        matrixIn=calcUnweightedSuperMatrix(model)
    node_names=nodeNameList(model)

    destinationFile='model_tempFile.png'

    for nodeName in args:
        points=sensitivityCellSupermatrixShort(model,clusterName,nodeName,verbal)

        cluster_obj=model.getClusterObjByName(clusterName)
        if(cluster_obj !=-1):
            nodesCluster=len(cluster_obj.nodes)
        #split resulting array to vectors
        else:
            nodesCluster=[]
            print("Cluster not found:", clusterName)
            return
        nodeY=[]
        x = reqLib.np.linspace(0, 1, 11)
        # print(x)
        fig,ax = reqLib.plt.subplots()
        for j in range (0,nodesCluster):
            
            pointsY=[]
            for i in range(0,11):
                pointsY.append(points[i][j])
            nodeY.append(pointsY)
            ax.plot(x, pointsY,label=cluster_obj.nodes[j].name)
        # Find the intersection points
        cross_points = []
        for k in range(nodesCluster):
            for l in range(k+1,nodesCluster):
                # print("k,l",k,l)
                p_points=[]
                q_points=[]
                for i in range(11):
                    p_points.append((x[i], nodeY[k][i]))
                    q_points.append((x[i], nodeY[l][i]))
                    # print((x[i], nodeY[k][i]))
                    # print((x[i], nodeY[l][i]))
                p=reqLib.LineString(p_points)
                q=reqLib.LineString(q_points)
                intersection = p.intersection(q)
                # Check if intersection point(s) is iterable
                # Check if there are intersection points
                
                if intersection:
                    # If multiple intersection points, iterate through them
                    if hasattr(intersection, '__iter__'):
                        for geom in intersection.geoms:
                            if isinstance(geom, reqLib.Point):
                                cross_points.append((round(geom.x, 3), round(geom.y, 3)))
                            elif isinstance(geom, reqLib.MultiPoint):
                                for point in geom:
                                    cross_points.append((round(point.x, 3), round(point.y, 3)))
                            # Add more elif cases for other geometry types if needed
                    else:  # Single intersection point
                        if isinstance(intersection, reqLib.Point):
                            cross_points.append((round(intersection.x, 3), round(intersection.y, 3)))
                            print(f'x: {round(intersection.x, 3)}, y: {round(intersection.y, 3)}')
                
        # print("crosspoints")
        # print(cross_points)
        # Add an annotation at each intersection point for point in cross_points:
        for point in cross_points: 
            txt="("+str(point[0])+" , "+str(point[1])+")"
            reqLib.plt.annotate(txt, xy=point, xytext=(point[0], point[1]+.1),
                    arrowprops=dict(facecolor='red', shrink=0.05))

        reqLib.plt.xlabel(nodeName)
        reqLib.plt.xlim(0, 1)
        reqLib.plt.ylim(0, 1)
        ticks = reqLib.np.arange(0, 1, 0.1)
        reqLib.plt.yticks(ticks)
        reqLib.plt.xticks(ticks)
        reqLib.plt.grid(linestyle='--', linewidth=.5)
        reqLib.plt.legend()

        if(filepath!="show"):

            reqLib.plt.savefig(destinationFile)
            sheet_name='Sens_'+nodeName
            with reqLib.pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:  
                workbook  = writer.book
                try:
                    writer.book.get_sheet_by_name(sheet_name)
                    sheet_exists = True
                except KeyError:
                    sheet_exists = False

                if not sheet_exists:
                    workbook.create_sheet(sheet_name)
                active = workbook[sheet_name]
                active.add_image(reqLib.Image(destinationFile),'A1')
                # workbook.save(filepath)
        

            # Check if the file destinationFile exists
            if reqLib.os.path.exists(destinationFile):
                # Delete the file
                reqLib.os.remove(destinationFile)

            else:
                print(destinationFile)
        else:
            reqLib.plt.show()

