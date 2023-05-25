import numpy as np
import pandas as pd


class cleaning:

    def __init__(self,data):
        self.data=data
        return
    
    def type_data_miss (self,tp=1):
        self.tp=tp
        if tp==1:
            self.missing=np.equal(self.data,"?")
            
        elif tp==2:
            self.missing=np.equal(self.data[:,0],0)
            self.missing=np.transpose([np.transpose(self.missing)]*20)
        return self
    
    
    def column_clean(self,trsh=1):
        miss=[]
        for i in range(self.data.shape[1]):
            num_non=sum(self.missing[:,i])
            if num_non>trsh*self.data.shape[0]:
                miss.append(i)
        self.data=np.delete(self.data, miss, 1)
        self.missing=np.delete(self.missing, miss, 1)
        return self
    
    
    def row_clean(self,trsh=1):
        miss=[]
        for i in range(self.data.shape[0]):
            num_non=sum(self.missing[i,:])
            if num_non>trsh*self.data.shape[1]:
                miss.append(i)
        self.data=np.delete(self.data, miss, 0)
        self.missing=np.delete(self.missing, miss, 0)
        return self

    def data_derivation(self):
        perfect_data_rindex=[]
        incomplete_realdata_index=[]
        incomplete_predictdata_index=[]
        
        for i in range(self.data.shape[0]):
            num_non=sum(self.missing[i,:])
            if num_non==0:
                perfect_data_rindex.append(i)
            else:
                incomplete_realdata_index.append([])
                incomplete_predictdata_index.append([])
                for j in range(self.data.shape[1]):
                    if self.missing[i,j]:
                        incomplete_predictdata_index[len(incomplete_realdata_index)-1].append([i,j])
                    else:
                        incomplete_realdata_index[len(incomplete_realdata_index)-1].append([i,j])

        for i in range(len(incomplete_realdata_index)):
            dist_predict=[]
            for j in perfect_data_rindex:
                dist=0
                for k in range(len(incomplete_realdata_index[i])):
                    dist=dist+(float(self.data[j,incomplete_realdata_index[i][k][1]])-float(self.data[incomplete_realdata_index[i][k][0],incomplete_realdata_index[i][k][1]]))**2
                dist_predict.append(dist)
            m_list=min(dist_predict)
            ind=dist_predict.index(m_list)
            
            for l in range(len(incomplete_predictdata_index[i])):
                self.data[incomplete_predictdata_index[i][l][0],incomplete_predictdata_index[i][l][1]]=self.data[perfect_data_rindex[ind],incomplete_predictdata_index[i][l][1]]
        return self

    
    def best_correlation(self,num_f):
        variance=np.var(self.data.astype(float) , axis=0)
        columns=np.arange(self.data.shape[1])
        for i in range(0,len(variance)):
            if variance[i]==0:
                self.data=np.delete(self.data, columns[i], 1)
        data_df=pd.DataFrame(self.data).astype(float)
        data_correlation=data_df.corr(method='pearson').abs()
        if self.tp==1:
            clm=0
        else:
            clm=data_df.shape[1]-1
        target_correlation=data_correlation[clm]
        target_correlation=target_correlation.sort_values(ascending=False)
        data_indices=target_correlation.index
        cleaning_indices=[]
        cleaning_indices.append([data_indices[1],target_correlation[data_indices[1]]])
        
        for f in range(1,num_f):
            corr=0
            for i in range(len(cleaning_indices)+1,len(target_correlation)):
                c_test=1
                aa=target_correlation[data_indices[i]]
                for j in range(len(cleaning_indices)):
                    dc=data_correlation[data_indices[i]][cleaning_indices[j][0]]
                    c_test=c_test*(1-dc)
                c_test=c_test*aa
                if c_test>corr:
                    corr=c_test
                    ind=data_indices[i]
            cleaning_indices.append([ind,corr])
        df=pd.DataFrame(cleaning_indices)
        df=df.sort_values(by=[1])
        self.data=np.delete(self.data, df[0][0:data_indices.size-num_f], 1)
        return self
    
    def give_data(self):
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.data[i][j]=float(self.data[i][j])
        return self.data