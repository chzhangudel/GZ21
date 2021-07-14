# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:28:27 2019

@author: Arthur
"""
import logging
import torch
import mlflow
from models.utils import load_model_cls, pickle_artifact

MODEL_RUN_ID = 'dc74cea68a7f4c7e98f9228649a97135'

print('To load the net from the paper, use the function load_paper_net().')

def load_paper_net():
    """
        Load the neural network from the paper
    """
    model_module_name = 'models1'
    model_cls_name = 'FullyCNN'
    model_cls = load_model_cls(model_module_name, model_cls_name)
    net = model_cls(2, 4)
    transformation = pickle_artifact(MODEL_RUN_ID, 'models/transformation')
    net.final_transformation = transformation

    # Load parameters of pre-trained model
    logging.info('Loading the neural net parameters')
    client = mlflow.tracking.MlflowClient()
    model_file = client.download_artifacts(MODEL_RUN_ID,
                                           'models/trained_model.pth')
    net.load_state_dict(torch.load(model_file))
    print(net)
    return net
