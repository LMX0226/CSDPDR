# CSDPDR
Drug Repositioning (DR), as an innovative drug development strategy, significantly reduces the cost and time of drug development by finding new indications for approved drugs. Currently, most methods focus on excavate information from the overall drug-disease network, but tend to ignore critical information in the local details between nodes. This study proposes the CSDPDR model, a two-branch graph neural network that fuses global and local information, aiming to improve the accuracy and efficiency of drug repositioning. With the global branching graph attention network (GAT) and subgraph branching Top-K pooling module, the model can capture both large-scale structural patterns and fine-grained local information. In addition, our approach effectively solves the graph sparsity problem through meta-path-based network enhancement and confidence filtering. The comparative experiments on two benchmark datasets demonstrate that the CSDPDR model significantly outperforms several baseline methods. Case studies of Alzheimer's disease and breast neoplasms further demonstrate the advantages of the model in practical applications.


pythorch ==2.0.1
cuda ==11.8
python 3.8
dgl 1.1.2+cu118
