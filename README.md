# Oil-gas-water three-phase flow process monitoring by Ag-MRDCVA
Source code of Ag-MRDCVA on oil-gas-water three-phase flow dataset.
The dataset is obtained through multiphase flow experiment at Tianjin Key Laboratory of Process Measurement and Control at Tianjin University.

The details of the model can be found in    
 [L. H. Li, et al. Manifold regularized deep canonical variate analysis with interpretable
attribute guidance for three-phase flow process monitoring, ESWA, 251, 124015, 2024.](https://doi.org/10.1016/j.eswa.2024.124015)

#### Fast execution in command line:  
python3 AgMRDCVA.py      

#### Results Example: 
Test case: data_ogw_test  
================= AgMRDCVA Training =================  
Epoch 0 | Total_loss: 4.6838  
Epoch 10 | Total_loss: 1.4853  
Epoch 20 | Total_loss: 1.4644  
Epoch 30 | Total_loss: 1.4428  
Epoch 40 | Total_loss: 1.4393  
......  
================= AgMRDCVA Testing =================  
Identification for typical flow states  
Overall Accuracy: 0.9174  
Accuracy for Class 0: 1.0000  
Accuracy for Class 1: 0.6550  
Accuracy for Class 2: 0.9350  
Accuracy for Class 3: 0.9700  
Accuracy for Class 4: 0.9600  
Accuracy for Class 5: 0.9550  
Accuracy for Class 6: 0.9800  
Accuracy for Class 7: 0.9000  
Accuracy for Class 8: 0.8400  
Accuracy for Class 9: 0.9945  


#### All rights reserved, citing the following papers are required for reference:   
[1] L. H. Li, et al. Manifold regularized deep canonical variate analysis with interpretable
attribute guidance for three-phase flow process monitoring, ESWA, 251, 124015, 2024.
