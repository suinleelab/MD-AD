import h5py
import pandas as pd
import numpy as np
from models import multitask_mlp
import keras
from keras.callbacks import CSVLogger
from keras import backend as K
import gc
from sklearn.model_selection import ParameterGrid
import os

class DataValidationExperiment:
    def __init__(self, fold_idx, train_datasets, test_datasets):
        self.fold_idx = fold_idx
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

        X_train, X_valid, y_train, y_valid = self.load_data_for_fold(fold_idx)
        self.X_train = X_train
        self.X_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid

    def load_data_for_fold(self, fold_idx):
        # Leaving this hardcoded for now.  I assume it won't change anytime soon (or ever)
        phenotypes = ["CERAD", "PLAQUES", "ABETA_IHC", "BRAAK", "TANGLES", "TAU_IHC"]
        num_components = 500
        num_cats = {"CERAD": 4, "BRAAK": 6}
        data_form = "ACT_MSBBRNA_ROSMAP_PCASplit"
        path_to_folders = "/../../../projects/leelab2/data/AD_DATA/Nicasia/processed/combined_files/prescience_data/"

        path_to_split_data = path_to_folders + data_form

        # ------------ LOAD DATA ---------------------------------------------------- #
        with h5py.File(path_to_split_data + "/" + str(fold_idx) + ".h5", 'r') as hf:
            if "PCA" in data_form or "KMeans" in data_form:
                X_train = hf["X_train_transformed"][:, :num_components].astype(np.float64)
                X_valid = hf["X_valid_transformed"][:, :num_components].astype(np.float64)
            else:
                X_train = hf["X_train"][:].astype(np.float64)
                X_valid = hf["X_valid"][:].astype(np.float64)
            labels_train = hf["y_train"][:]
            labels_valid = hf["y_valid"][:]
            labels_names = hf["labels_names"][:]

        # ------------ PROCESS DATA FOR MODEL: TRAIN/TEST SETS + LABELS----------------- #
        labels_df_train = pd.DataFrame(labels_train, columns=labels_names.astype(str))
        labels_df_valid = pd.DataFrame(labels_valid, columns=labels_names.astype(str))

        training_points_in_datasets = labels_df_train.filename.str.decode("utf-8").isin(self.train_datasets)
        X_train = X_train[training_points_in_datasets]
        labels_df_train = labels_df_train[training_points_in_datasets]

        valid_points_in_datasets = labels_df_valid.filename.str.decode("utf-8").isin(self.test_datasets)
        X_valid = X_valid[valid_points_in_datasets]

        labels_df_valid = labels_df_valid[valid_points_in_datasets]

        shuffle_idx_train = np.random.permutation(range(len(labels_df_train)))
        shuffle_idx_valid = np.random.permutation(range(len(labels_df_valid)))

        X_train = X_train[shuffle_idx_train]
        X_valid = X_valid[shuffle_idx_valid]

        labels_df_train = labels_df_train.reset_index(drop=True).loc[shuffle_idx_train]
        labels_df_valid = labels_df_valid.reset_index(drop=True).loc[shuffle_idx_valid]

        y_train = {}
        y_valid = {}
        for phen in phenotypes:
            if phen in num_cats.keys():
                y_train[phen] = labels_df_train[phen].astype(float).values / (num_cats[phen] - 1)
                y_valid[phen] = labels_df_valid[phen].astype(float).values / (num_cats[phen] - 1)
            else:
                y_train[phen] = labels_df_train[phen].astype(float).values
                y_valid[phen] = labels_df_valid[phen].astype(float).values

        return X_train, X_valid, y_train, y_valid

    def run_experiment(self):

        hyperparams = {"epochs": [200],
                       "nonlinearity": ["relu"],
                       "hidden_sizes_shared": [[500, 100]],
                       "hidden_sizes_separate": [[50, 10]],
                       "dropout": [.1],
                       "k_reg": [.00001, .001],
                       "learning_rate": [.0001, .001],
                       "loss_weights": [[1, 1]],
                       "grad_clip_norm": [.01, .1],
                       "batch_size": [20]}

        hy_dict_list = list(ParameterGrid(hyperparams))

        for hy_dict in hy_dict_list:
            num_epochs = hy_dict["epochs"]
            nonlinearity = hy_dict["nonlinearity"]
            hidden_sizes_shared = hy_dict["hidden_sizes_shared"]
            hidden_sizes_separate = hy_dict["hidden_sizes_separate"]
            dropout = hy_dict["dropout"]
            k_reg = hy_dict["k_reg"]
            learning_rate = hy_dict["learning_rate"]
            loss_weights = hy_dict["loss_weights"]
            grad_clip_norm = hy_dict["grad_clip_norm"]
            batch_size = hy_dict["batch_size"]

            title = str(self.train_datasets)
            #title = "%d_%s_%s_%s_%f_%f_%f_%s_%f_%d" % (
            #    num_epochs, nonlinearity, str(hidden_sizes_shared), str(hidden_sizes_separate),
            #    dropout, k_reg, learning_rate, str(loss_weights), grad_clip_norm, batch_size
            #)

            path_to_results = "MODEL_OUTPUTS/results/md-ad_data_validation/"
            path_to_preds = "MODEL_OUTPUTS/predictions/md-ad_data_validation/"
            path_to_models = "MODEL_OUTPUTS/models/md-ad_data_validation/"

            data_form = "ACT_MSBBRNA_ROSMAP_PCASplit"
            model = multitask_mlp(self.X_train, hy_dict)

            # https://stackoverflow.com/questions/36895627/python-keras-creating-a-callback-with-one-prediction-for-each-epoch
            X_valid = self.X_valid
            class prediction_history(keras.callbacks.Callback):
                def __init__(self):
                    self.predhis = []

                def on_epoch_end(self, epoch, logs={}):
                    self.predhis.append(model.predict(X_valid))

            predictions = prediction_history()

            res_dest = path_to_results + "/" + data_form + "/" + title + "/"
            if not os.path.isdir(res_dest):
                os.makedirs(res_dest)

            preds_dest = path_to_preds + "/" + data_form + "/" + title + "/"
            if not os.path.isdir(preds_dest):
                os.makedirs(preds_dest)

            modelpath = path_to_models + data_form + "/" + title + "/" + str(self.fold_idx) + "/"
            if not os.path.isdir(modelpath):
                os.makedirs(modelpath)

            csv_logger = CSVLogger(res_dest + '%d.log' % self.fold_idx)

            print("Fitting model")
            History = model.fit(x={'main_input': self.X_train},
                                y={'BRAAK_out': self.y_train["BRAAK"], 'CERAD_out': self.y_train["CERAD"],
                                   'PLAQUES_out': self.y_train["PLAQUES"], 'TANGLES_out': self.y_train["TANGLES"],
                                   "ABETA_IHC_out": self.y_train["ABETA_IHC"], "TAU_IHC_out": self.y_train["TAU_IHC"]},
                                validation_data=({'main_input': self.X_valid},
                                                 {'BRAAK_out': self.y_valid["BRAAK"], 'CERAD_out': self.y_valid["CERAD"],
                                                  'PLAQUES_out': self.y_valid["PLAQUES"], 'TANGLES_out': self.y_valid["TANGLES"],
                                                  "ABETA_IHC_out": self.y_valid["ABETA_IHC"],
                                                  "TAU_IHC_out": self.y_valid["TAU_IHC"]}),

                                verbose=0, epochs=num_epochs, batch_size=batch_size, callbacks=[csv_logger, predictions,
                                                                                                keras.callbacks.ModelCheckpoint(
                                                                                                    modelpath + "{epoch:02d}.hdf5",
                                                                                                    monitor='val_loss',
                                                                                                    verbose=0,
                                                                                                    save_best_only=False,
                                                                                                    save_weights_only=False,
                                                                                                    mode='auto',
                                                                                                    period=100)])

            # SAVE PREDICTIONS
            with h5py.File(preds_dest + "%d.h5" % self.fold_idx, 'w') as hf:
                # loop through epochs -- one group is made per epoch
                for i, ep in enumerate(predictions.predhis):
                    # within each group created for each epoch, we save a dataset of predictions for the validation set
                    for j, phenotype in enumerate(["BRAAK", "CERAD", "PLAQUES", "TANGLES",
                                                   "ABETA_IHC", "TAU_IHC"]):

                        if "/%s/%s" % (str(i), phenotype) in hf:
                            del hf["/%s/%s" % (str(i), phenotype)]
                        hf.create_dataset("/%s/%s" % (str(i), phenotype), data=predictions.predhis[i][j],
                                          dtype=np.float32)
                # save true values to /y_true/phenotype
                for phenotype in ["BRAAK", "CERAD", "PLAQUES", "TANGLES",
                                  "ABETA_IHC", "TAU_IHC"]:
                    if "/y_true/" + phenotype in hf:
                        del hf["/y_true/" + phenotype]
                    hf.create_dataset("/y_true/" + phenotype, data=self.y_valid[phenotype], dtype=np.float32)


            K.clear_session()
            gc.collect()
            break