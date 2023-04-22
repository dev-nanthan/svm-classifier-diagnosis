# Import necessary libraries
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from pathlib import Path


def diagnoseDAT(Xtest, data_dir):

  # Load the Data
  #############################################
  p_train_sNC = os.path.join(data_dir, "train.fdg_pet.sNC.csv") 
  p_train_sDAT = os.path.join(data_dir, "train.fdg_pet.sDAT.csv") 
  p_test_sNC = os.path.join(data_dir, "test.fdg_pet.sNC.csv") 
  p_test_sDAT = os.path.join(data_dir, "test.fdg_pet.sDAT.csv")

  # Set the feature names
  feature_nm = ['ctx-lh-inferiorparietal',
                'ctx-lh-inferiortemporal', 
                'ctx-lh-isthmuscingulate', 
                'ctx-lh-middletemporal',
                'ctx-lh-posteriorcingulate', 
                'ctx-lh-precuneus',
                'ctx-rh-isthmuscingulate',
                'ctx-rh-posteriorcingulate', 
                'ctx-rh-inferiorparietal', 
                'ctx-rh-middletemporal',
                'ctx-rh-precuneus', 
                'ctx-rh-inferiortemporal',
                'ctx-lh-entorhinal',
                'ctx-lh-supramarginal']

  
  train_sNC = pd.read_csv(p_train_sNC, names=feature_nm, header=None)
  train_sDAT = pd.read_csv(p_train_sDAT, names=feature_nm, header=None)
  test_sNC = pd.read_csv(p_test_sNC, names=feature_nm, header=None)
  test_sDAT = pd.read_csv(p_test_sDAT, names=feature_nm, header=None)

  # Extract the Data
  #############################################
  # Extract the Design Matrix of 14 features from the Data Set

  X_train = np.concatenate([train_sNC, train_sDAT])
  Y_train = np.concatenate([np.zeros(len(train_sNC)), np.ones(len(train_sDAT))])
  X_test = np.concatenate([test_sNC, test_sDAT])
  Y_test = np.concatenate([np.zeros(len(test_sNC)), np.ones(len(test_sDAT))])

  # Find the best SVM hyperparameters using grid search with cross-validation
  # ############################################################################ 
  param_grid = {'C': [0.1, 1, 5, 10, 100, 200], 'gamma': [0.001, 0.005, 0.01, 0.1, 1, 10]}
  svm = SVC(kernel='rbf')
  svm_cv = GridSearchCV(svm, param_grid, cv=5)
  svm_cv.fit(X_train, Y_train)
  best_C = svm_cv.best_params_['C']
  best_gamma = svm_cv.best_params_['gamma']
  print("Best SVM hyperparameters: C = {}, gamma = {}".format(best_C, best_gamma))
  
  # Create the Final SVM classifier model using "Best" Hyper-Parameters
  # ############################################################################ 
  svm_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma)

  # Combine Training data and the Validation/Test Data to increase training data
  X_Comb = np.concatenate([X_train, X_test])
  Y_Comb = np.concatenate([Y_train, Y_test])

  svm_final.fit(X_Comb, Y_Comb)

 
  # Make predictions on the Blinded test data
  y_pred = svm_final.predict(Xtest)

  # Return the predictions on the Blinded Test Data
  return y_pred
