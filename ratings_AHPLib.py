from AhpAnpLib import functions_AHPLib as reqLib
from AhpAnpLib import inputs_AHPLib as cdf_inp
from AhpAnpLib import structs_AHPLib as cdf_str
from AhpAnpLib import calcs_AHPLib as cdf_calc
# import functions_AHPLib as reqLib
# import inputs_AHPLib as cdf_inp
# import structs_AHPLib as cdf_str
# import calcs_AHPLib as cdf_calc
class RatScale:
    id_generator = reqLib.itertools.count(0)
    def __init__(self,scale_name):
        self.name = scale_name
        self.scaleID = next(self.id_generator)
        #members are [name,ideal priority] or [name,1] if no values are given assuming all equal
        self.members=[]
        #val matrix is the pc matrix that would give the ideal priority vector
        self.val_mat=[]

    def defineScaleByValue(self,mat_values=None,verbal=False,*all_values):
        valuesVector=[]
        
        for value in all_values:
            if isinstance(value, list):
                valuesVector.append(value[1])
            else:
                valuesVector.append(1)
        
        valuesIdealVector=cdf_calc.idealVector(valuesVector)
        i=0
        for value in all_values:
            if isinstance(value, list):
                value[1]=valuesIdealVector[i]
                self.members.append(value)
            else:
                self.members.append([value,1])
            i+=1
        if(mat_values is None):
            self.val_mat=cdf_inp.convDirect2PairwiseMatrix(valuesIdealVector)
        else:
            self.val_mat= reqLib.cp.deepcopy(mat_values)

        if(verbal):
            print(all_values)
            print(valuesVector)
            print(self.val_mat)
    
    def __repr__(self):
        labels=[]
        for value in self.members:
            labels.append(value[0])
        scaleDesc="\n-------------------Rating model scale---------\nName:"+ self.name+"\n"+reqLib.tabulate(self.members, headers=['Member name', 'Value'], tablefmt='orgtbl')+"\n--------------------------------------------\n"+reqLib.tabulate(self.val_mat, headers=labels, tablefmt='orgtbl'+"\n")
        return scaleDesc

    def getmIndexBymName(self,mName,verbal=False):
        if verbal: print(f"Checking:{mName}")
        for i,member in enumerate(self.members):
            if verbal: print(f"??={member[0]}")
            if mName==member[0]:
                if verbal:
                    print (f"Found in pos {i}")
                return i
        return -1
    def getmValueBymName(self,mName,verbal=False):
        # print(mName)
        index=self.getmIndexBymName(mName,verbal)
        
        if index!=-1:

            pr_vector=cdf_calc.priorityVector(self.val_mat)
            id_vect=cdf_calc.idealVector(pr_vector)
            if verbal:
                    print (f"Scale: {self.val_mat}\n Priorities {pr_vector} \n Ideal {id_vect}")
            return(id_vect[index])
        else:
            return -1 

class RatStruct:
    def __init__(self,parent):
        self.parent=parent
        #existing model nodes to be used for the ratings model columns (criteria) 
        #each model can have only one rat model attached to it
        self.ratCriteria=[]
        #existing model nodes to be used for the ratings model's rows (alternatives)
        self.ratAlternatives=[]
        #list of ratScales available to this ratings model 
        self.ratScales=[]
        #assignment of scale to criterion
        self.critScaleCombo=[]
        #the ratings table containing the indices of the selected scale items
        self.ratMatrix=[]
        #the priority vector to be used for the calculations - you get it from the limiting priorities re-normalized after removing the unused components
        self.ratCritPriorities=[]

    def __repr__(self):
        return "\n Criteria"+str(self.ratCriteria)+"\n Alternatives to be rated:"+str(self.ratAlternatives)+"\n Scales available to use"+str(self.ratScales)+"\n Criteria and selected scales for each of them:"+str(self.critScaleCombo)
    
    def addCriteriaByVar(self,*args):
        for node in args:
            self.ratCriteria.append(node)
        
    def addCriteriaByName(self,*args):
        for nodeName in args:
            self.ratCriteria.append(self.parent.getNodeObjByName(nodeName))
    
    def addAlternativesByVar(self,*args):
        for node in args:
            self.ratAlternatives.append(node)
        
    def addAlternativesByName(self,*args):
        #if the node does not exist a new node will be created and added to the ratingAlternatives cluster
        for nodeName in args:
            theNode=self.parent.getNodeObjByName(nodeName)
            if(theNode!=-1):
                self.ratAlternatives.append(theNode)
            else:
                #the node does not exist so let's create it
                theAltCluster=self.parent.getClusterIDByName("Alternatives")
                if(theAltCluster==-1):
                    new_nd=cdf_str.Node(nodeName,1+self.parent.getMaxNodeOrder())
                    cl_index=self.parent.getClusterIndexByName("ratingAlternatives")
                    if(cl_index==-1):
                        new_cl=cdf_str.Cluster("ratingAlternatives",1+self.parent.getMaxClusterOrder())
                        new_cl.addNode2Cluster(new_nd)
                        self.parent.addCluster2Model(new_cl)
                    else:
                        self.parent.clusters[cl_index].addNode2Cluster(new_nd)
                self.ratAlternatives.append(new_nd)

    def addScaleByVar(self,*args):
        for scale in args:
            self.ratScales.append(scale)
            # print(self.ratScales)

    def scaleExists(self,name):
        for indx,item in enumerate(self.ratScales):
            if (item.name==name):
                return indx
        return -1
    
    def criterionExists(self,name):
        for indx,item in enumerate(self.ratCriteria):
            if (item.name==name):
                return indx
        return -1
    
    def getScaleObjByName(self,name):

        for scale in self.ratScales:
            if (scale.name==name):
                return scale
        return -1
    def getRatCritObjByName(self,name):
        for item in self.ratCriteria:
            if (item.name==name):
                return item
        return -1
    
    def assignScale2CriterionByName(self,criterionName,scaleName,verbal=False):
        #check if scale exists in the ratings model
        scale_index=self.scaleExists(scaleName)
        #check if criterion belongs to this ratings model
        crit_index=self.criterionExists(criterionName)

        #cannot have 2 criteria connected to the same scale object so if the same create a copy
        found = any(scaleName in tuple_ for tuple_ in self.critScaleCombo)
        #add scale, criterion tuple to the ratings model
        if(found):
            scale2copy=reqLib.cp.deepcopy(self.getScaleObjByName(scaleName))
            new_name=scaleName+"_cp"
            found_nm = any(new_name in scale.name for scale in self.ratScales)
            while(found_nm):
                new_name+="_cp"
            scale2copy.name=new_name
            scale2copy.ID=next(scale2copy.id_generator)
            self.addScaleByVar(scale2copy)
            self.critScaleCombo.append([criterionName,new_name])
        else:
            self.critScaleCombo.append([criterionName,scaleName])
        if (verbal):
            print(f"{scaleName} found in position {scale_index}\n")
            print(f"{criterionName} found in position {crit_index}\n")
            print(f"{self.critScaleCombo} updated\n")

    def getScaleOfCriterion(self, criterion):
        if(self.criterionExists(criterion.name)!=-1):
            
            for pair in self.critScaleCombo:
                if (pair[0]==criterion.name):
                    scale_name=pair[1]
                    return self.getScaleObjByName(scale_name)
        return -1


    def updateScaleMatrix(self,scale_name,new_val_matrix):
        #update the scale matrix and member values with what you just read
        for ind,scale in enumerate(self.ratScales):
            if scale.name==scale_name:
                self.ratScales[ind].val_mat=new_val_matrix
                pr_vector=cdf_calc.priorityVector(new_val_matrix)
                id_vect=cdf_calc.idealVector(pr_vector)
                for i,member in enumerate(scale.members):
                    member[1]=id_vect[i]
                # print(scale.members)