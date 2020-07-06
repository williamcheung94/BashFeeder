# MLP for Pima Indians Dataset with grid search via sklearn
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten, BatchNormalization, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

leaky_relu = tf.nn.leaky_relu
# Function to create model, required for KerasClassifier
'''
def create_model(learn_rate =.001, activation="relu", init='glorot_uniform', kernel_regularizer="l1", dropout=0, input_shape= (20,1)):
    batch_size = 100
    # create model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(280, 8, padding = "same", kernel_initializer=init, activation=activation, kernel_regularizer=kernel_regularizer, input_shape=(20,1,1)))  # input_dim=(20, )
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, 8, padding = "same", kernel_initializer=init, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 8, padding = "same", kernel_initializer=init, activation=activation, kernel_regularizer=kernel_regularizer))
    model.add(Flatten())
    #model.add(Dropout(dropout))
    model.add(Dense(1, kernel_initializer=init, activation=activation, kernel_regularizer=kernel_regularizer))


    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr = learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
'''


def create_model(learn_rate=.001, activation="relu", init='glorot_uniform', kernel_regularizer="l1",
                 inputshape=(None, 40)):
    # create model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(32, 8, padding="same", input_shape=(inputshape, 1), kernel_initializer=init,
                     kernel_regularizer=regularizers.l2(0.01)))  # input_dim=(20, )
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="relu"))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_Mel(learn_rate=.001, activation="relu", init='glorot_uniform', kernel_regularizer="l1",
                     inputshape=(None, 128)):
    # create model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(32, 8, padding="same", input_shape=(inputshape, 1), kernel_initializer=init,
                     kernel_regularizer=regularizers.l2(0.01)))  # input_dim=(20, )
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="relu"))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_RMS(learn_rate=.001, activation="relu", init='glorot_uniform', kernel_regularizer="l1",
                     inputshape=(None, 20)):
    # create model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(32, 8, padding="same", input_shape=(inputshape, 1), kernel_initializer=init,
                     kernel_regularizer=regularizers.l2(0.01)))  # input_dim=(20, )
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="relu"))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_Chromashift(learn_rate=.001, activation="relu", init='glorot_uniform', kernel_regularizer="l1",
                             inputshape=(None, 12)):
    # create model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(32, 8, padding="same", input_shape=(inputshape, 1), kernel_initializer=init,
                     kernel_regularizer=regularizers.l2(0.01)))  # input_dim=(20, )
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="relu"))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_specCont(learn_rate=.001, activation="relu", init='glorot_uniform', kernel_regularizer="l1",
                          inputshape=(None, 10)):
    # create model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(32, 8, padding="same", input_shape=(inputshape, 1), kernel_initializer=init,
                     kernel_regularizer=regularizers.l2(0.01)))  # input_dim=(20, )
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="relu"))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_tonnetz(learn_rate=.001, activation="relu", init='glorot_uniform', kernel_regularizer="l1",
                         inputshape=(None, 6)):
    # create model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(32, 8, padding="same", input_shape=(inputshape, 1), kernel_initializer=init,
                     kernel_regularizer=regularizers.l2(0.01)))  # input_dim=(20, )
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="relu"))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_LPCRPLP(learn_rate=.001, activation="relu", init='glorot_uniform', kernel_regularizer="l1",
                         inputshape=(None, 13)):
    # create model
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Conv1D(32, 8, padding="same", input_shape=(inputshape, 1), kernel_initializer=init,
                     kernel_regularizer=regularizers.l2(0.01)))  # input_dim=(20, )
    model.add(Activation(activation))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01), activation="relu"))

    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


tf.compat.v1.enable_eager_execution()
'''
Data Pull
Here I pull the train test data from excel files.
'''

File = "S_10pData_withNoise_6_27_1.63sec"

File_types = ["GFCC"]
# ["GFCC", "LFCC","BFCC", "NGCC", "LPC", "RPLP", "chromashift", "melspect", "chromaCqt", "RMS", "specCont", "tonnetz", "MFCC", "delta", "deltadelta"]
for item in File_types:
    File_Name = File + "_" + item + ".xlsx"

    df = pd.read_excel(File_Name, "Sheet1", header=0, usecols="B:HAK")

    print(File_Name)

    people = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sounds = [1, 2, 3, 4, 5, 6]
    scores_df = pd.DataFrame(None, columns=["Target ID", "Sound", "Model", "TP", "FN", "FP", "TN", "best params"])
    validation_df = pd.DataFrame(None, columns=["Target ID", "Sound", "Model", "means", "stds", "params"])
    feature_count = 20

    df = df.sample(frac=1)

    best_model = None
    best_model_performance = 0
    best_model_summary = None

    for column in df.columns:
        if ((column != "Person ID") & (column != "Sound ID")):
            df[column] = df[column] / df[column].max()

    pd.options.mode.use_inf_as_na = True
    df = df.replace([np.inf, -np.inf], np.nan)
    non_null_column = df.isnull().sum()[df.isnull().sum() == 0].index
    df = df[non_null_column]

    # for person in people:
    for person in [1, 2, 3, 4]:
        validUser = person
        # for sound in sounds:
        for sound in sounds:

            k_values = [30]

            for k_value in k_values:
                print("validUser is " + str(validUser))
                print("sound left out" + str(sound))
                print("K features" + str(k_value))
                # print(len(df.loc[(df["Person ID"] == validUser) & (df["Sound ID"] <= sound)])//(len(people)-2))
                traindf = df.loc[(df["Person ID"] == validUser) & (df["Sound ID"] != sound)]
                for person in people:
                    if (person != validUser):
                        # for i in range(0,int(len(sounds)//(len(people)-1))):
                        traindf = pd.concat(
                            [traindf, df.loc[(df["Person ID"] == person) & (df["Sound ID"] != sound)][:len(
                                df.loc[(df["Person ID"] == validUser) & (df["Sound ID"] != sound)]) // (len(
                                people) - 1)]])

                testdf = df.loc[(df["Person ID"] == validUser) & (df["Sound ID"] == sound)]
                for person in people:
                    if (person != validUser):
                        testdf = pd.concat(
                            [testdf, df.loc[(df["Person ID"] == person) & (df["Sound ID"] == sound)][:len(
                                df.loc[(df["Person ID"] == validUser) & (df["Sound ID"] == sound)]) // (len(
                                people) - 1)]])

                traindf = traindf.sample(frac=1)
                testdf = testdf.sample(frac=1)

                X_train = traindf.drop(columns=["Person ID", "Sound ID"])
                X_test = testdf.drop(columns=["Person ID", "Sound ID"])

                Y_train = traindf["Person ID"]
                Y_test = testdf["Person ID"]
                for person in people:
                    if (person != validUser):
                        # print(person)
                        Y_train = Y_train.replace({person: 0})
                        Y_test = Y_test.replace({person: 0})

                for person in people:
                    if (person == validUser):
                        # print(person)
                        Y_train = Y_train.replace({person: 1})
                        Y_test = Y_test.replace({person: 1})

                print("Training Size: ", len(X_train), " balance of: ", Y_train.sum(), ":",
                      len(Y_train) - Y_train.sum(), \
                      "Training Size: ", len(X_test), " balance of : ", Y_test.sum(), ":", len(Y_test) - Y_test.sum())
                '''
                Feature Selection
                Here I attempt to extract the most effective features. I try two feature selection methods: SelectKBest and SelectFromModel.

                Currently we are looking at the top 10 features but in the future I want to try 10 fold on training data to find best number
                of features to have.
                '''
                '''
                if ((str(item) == "chromashift") or (str(item) == "specCont") or (str(item) == "tonnetz")):
                    pass
                else:
                    # Layer one of feature selection using correlation
                    corr = X_train.corr()
                    absCorr = np.abs(corr)

                    one_minus_alpha = .9

                    for row in range(len(absCorr)):
                        for column in range(len(absCorr.iloc[row])):
                            # print("row:", row, " column:", column)
                            if (row == column):
                                continue
                            elif (absCorr.iloc[row][column] >= one_minus_alpha):
                                # print(absCorr.iloc[row][column])
                                try:
                                    X_train = X_train.drop(labels=absCorr.columns[column], axis=1)
                                    X_test =  X_test.drop(labels=absCorr.columns[column], axis=1)
                                except:
                                    pass
                                    # Exception is usually when the item is deleted already but we find another correlation.
                                    # We ignore this because we don't want to delete the df that we are itering through.

                # Layer two of feature selection using PCA
                if (len(X_train.columns) < feature_count):
                    feature_count = len(X_train.columns)
                    print("too little features pass correlation PCA used is: ", feature_count)
                pca = PCA(n_components=feature_count)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)
                '''
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)
                print("shapes", X_train.shape, Y_train.shape)
                X_train = X_train.reshape(-1, X_train.shape[1], 1)
                # Y_train = np.array(Y_train)
                # Y_train = Y_train.reshape(-1,1,1)
                X_test = X_test.reshape(-1, X_train.shape[1], 1)

                print("final", X_train.shape, Y_train.shape)
                # Trainning_Set = (tf.data.Dataset.from_tensor_slices((tf.cast(X_train, tf.float32),
                #                                                     tf.cast(Y_train, tf.int32))))

                # Test_Set = (tf.data.Dataset.from_tensor_slices((tf.cast(X_test, tf.float32),
                #                                                tf.cast(Y_test, tf.int32))))
                # Trainning_Set = Trainning_Set.reshape((1,20,1))
                # print(Trainning_Set)
                # print(Test_Set)
                if (str(item) == "chromashift"):
                    model = KerasClassifier(build_fn=create_model_Chromashift, verbose=0)
                elif (str(item) == "specCont"):
                    model = KerasClassifier(build_fn=create_model_specCont, verbose=0)
                elif (str(item) == "tonnetz"):
                    model = KerasClassifier(build_fn=create_model_tonnetz, verbose=0)
                elif ((str(item) == "LPC") or (str(item) == "RPLP") or (str(item) == "BFCC")):
                    model = KerasClassifier(build_fn=create_model_LPCRPLP, verbose=0)
                elif (str(item) == "RMS"):
                    model = KerasClassifier(build_fn=create_model_RMS, verbose=0)
                elif (str(item) == "melspect"):
                    model = KerasClassifier(build_fn=create_model_Mel, verbose=0)
                else:
                    print(item == "chromashift")
                    model = KerasClassifier(build_fn=create_model, verbose=0)
                # grid search epochs, batch size and optimizer
                optimizers = ['adam']  # ['adam', "SGD", 'rmsprop', "Adadelta", "adagrad", "adamax", "Nadam", "Ftrl"]
                activation = ['relu']  # [leaky_relu, 'relu', 'sigmoid']
                init = ['normal']  # ['glorot_uniform', 'normal', 'uniform', 'zero']
                epochs = [100]  # [50, 100, 150]
                batches = [50, 75, 100]  # [25, 50, 75, 100]
                learn_rate = [.00001, .001, .1]
                regularizer = ["l2"]  # ["l1", "l2"]
                # dropout = [0.2, 0.4, 0.6] #[0.0, 0.2, 0.4, 0.6]
                param_grid = dict(activation=activation, epochs=epochs, batch_size=batches, learn_rate=learn_rate,
                                  init=init,
                                  kernel_regularizer=regularizer)  # , dropout=dropout)  # optimizer=optimizers, , momentum=momentum
                grid = GridSearchCV(estimator=model, param_grid=param_grid)
                grid_result = grid.fit(X_train, Y_train)
                # grid_results = grid_result.score(X_test, Y_test)
                # summarize results
                # print("Training: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                means = grid_result.cv_results_['mean_test_score']
                stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                    print("Trainning " + str(validUser) + " " + str(sound) + " %f (%f) with: %r" % (mean, stdev, param))
                    validation_info = [validUser, sound, "CNN (Adam)", means, stds, params]
                    validation_df = validation_df.append(pd.Series(validation_info, index=validation_df.columns),
                                                         ignore_index=True)

                pred_keras = grid_result.predict(X_test)
                matrix = confusion_matrix(Y_test.values.ravel(), pred_keras)
                TP, FN, FP, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
                score_info = [validUser, sound, "CNN", TP, FN, FP, TN, grid_result.best_params_]
                scores_df = scores_df.append(pd.Series(score_info, index=scores_df.columns), ignore_index=True)
                print("Testing " + str(validUser) + " " + str(sound) + " " + str(TP) + " " + str(FN) + " " + str(
                    FP) + " " + str(TN))
                print("Testing " + str(validUser) + " " + str(sound) + " ", grid_result.best_params_)
                '''
                model_to_save = grid_result
                #best_model.save_model()
                converter = tf.lite.TFLiteConverter.from_keras_model(model_to_save)
                tflite_model = converter.convert()
                with tf.io.gfile.GFile("model.tflite", "wb") as f:
                    f.write(tflite_model)
                '''
    print("breakpoint save for ", item)
    writer = pd.ExcelWriter("Kera_Results_for_128-64_" + item + "_6_27.xlsx", engine='xlsxwriter')
    scores_df.to_excel(writer, sheet_name='Sheet1')
    validation_df.to_excel(writer, sheet_name="sheet2")
    # writer.save()
    # print(scores_df)
