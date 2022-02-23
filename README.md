In the iPython Notebook from Sagemaker select the kernel "conda_mxnet_36".  
The file `urinary_diseases_191.ipynb` is a local implementation of the model using `Torch==1.9.1`. The file `urinary_diseases_110.ipynb` is also a local model but using `Torch==1.10`, but this file is in the project just for analysis, Sagemaker doesn't support it so it can't be deployed.

The torch version to be used in the project is 1.9.1, as this is supported in Sagemaker.   
In the iPython Notebook use  
    
    %pip install torch==1.9.1

This kernel already has these libraries installed, but I checked their version just for verification.

    pandas 1.1.5  
    sagemaker 2.72.1  
    numpy 1.19.5  

These other libraries are used in the iPython Notebooks, but not used in production code as they are just for visualization. They are also pre-installed.

    seaborn 0.11.1
    matplotlib 3.3.4
    sklearn 0.24.1

When calling `PyTorch` from `sagemaker.pytorch`, need to choose the correct `framework_version=1.9.1` and `py_version='py38'`

    from sagemaker.pytorch import PyTorch
    estimator = PyTorch(entry_point='train.py',
                        source_dir='source',
                        role=role,
                        framework_version='1.9.1',
                        py_version='py38',
                        train_instance_count=1,
                        train_instance_type='ml.c4.xlarge',
                        output_path=output_path,
                        sagemaker_session=sagemaker_session,
                        hyperparameters={
                            'input_features': INPUT_DIM,
                            'hidden_dim': HIDDEN_DIM,
                            'output_dim': OUTPUT_DIM,
                            'epochs': 300
                        })
    model = PyTorchModel(model_data=estimator.model_data,
                        role = role,
                        framework_version='1.9.1',
                        py_version='py38',
                        entry_point='predict.py',
                        source_dir='source')

The dataset is TAB delimited, UTF-16 and doesn't have headers.

    data_file = 'data/diagnosis.data'
    columns = ['Temperature of patient', 'Occurrence of nausea', 'Lumbar pain', 'Urine pushing (continuous need for urination)', 'Micturition pains', 'Burning of urethra, itch, swelling of urethra outlet', 'Inflammation of urinary bladder', 'Nephritis of renal pelvis origin']
    data_df = pd.read_csv(filepath_or_buffer=data_file, sep='\t', header=None, names=columns, encoding='utf-16')

The web page was deployed using S3 with static hosting enabled, the file `index.html` is provided. This S3 needs public access policy as in `aws.json`. The lambda function code is provided in `lambda_fn.py`. The API Gateway is deployed with CORS enabled.


### Credit
The dataset requires to give citation and credit:

    Citation Request:
    J.Czerniak, H.Zarzycki, Application of rough sets in the presumptive diagnosis of urinary system diseases, Artifical Inteligence and Security in Computing Systems, ACS'2002 9th International Conference Proceedings, Kluwer Academic Publishers,2003, pp. 41-51

Source: http://archive.ics.uci.edu/ml/datasets/Acute+Inflammations 
