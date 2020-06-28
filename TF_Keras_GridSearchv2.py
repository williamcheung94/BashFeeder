# MLP for Pima Indians Dataset with grid search via sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


# Function to create model, required for KerasClassifier
def create_model(learn_rate=0.01, init='glorot_uniform', input_dim=20):
    # create model
    activation = "relu"
    model = Sequential()
    model.add(Dense(30, input_dim=input_dim, kernel_initializer=init, activation=activation))  # input_dim=(20, )
    model.add(Dense(60, kernel_initializer=init, activation=activation))
    model.add(Dense(1, kernel_initializer=init, activation=activation))
    '''
    model.add(Dense(20, input_dim=20, kernel_initializer=init, activation=activation)) #input_dim=(20, )
    model.add(Dense(80, kernel_initializer=init, activation=activation))
    model.add(Dense(160, kernel_initializer=init, activation=activation))
    model.add(Dense(320, kernel_initializer=init, activation=activation))
    model.add(Dense(320, kernel_initializer=init, activation=activation))
    model.add(Dense(160, kernel_initializer=init, activation=activation))
    model.add(Dense(80, kernel_initializer=init, activation=activation))
    model.add(Dense(1, kernel_initializer=init, activation=activation))
    '''
    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_Chromashift(learn_rate=0.01, init='glorot_uniform', input_dim=12):
    # create model
    activation = "relu"
    model = Sequential()
    model.add(Dense(30, input_dim=input_dim, kernel_initializer=init, activation=activation))  # input_dim=(20, )
    model.add(Dense(60, kernel_initializer=init, activation=activation))
    model.add(Dense(1, kernel_initializer=init, activation=activation))
    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_specCont(learn_rate=0.01, init='glorot_uniform', input_dim=10):
    # create model
    activation = "relu"
    model = Sequential()
    model.add(Dense(30, input_dim=input_dim, kernel_initializer=init, activation=activation))  # input_dim=(20, )
    model.add(Dense(60, kernel_initializer=init, activation=activation))
    model.add(Dense(1, kernel_initializer=init, activation=activation))
    # Compile model
    optimizer = tf.keras.optimizers.Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def create_model_tonnetz(learn_rate=0.01, init='glorot_uniform', input_dim=6):
    # create model
    activation = "relu"
    model = Sequential()
    model.add(Dense(30, input_dim=input_dim, kernel_initializer=init, activation=activation))  # input_dim=(20, )
    model.add(Dense(60, kernel_initializer=init, activation=activation))
    model.add(Dense(1, kernel_initializer=init, activation=activation))
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

File = "S_10pData_withNoise_6_24_1.23sec"

File_types = ["specCont", "tonnetz"]
# "GFCC", "LFCC", "MFCC", "MSRCC", "NGCC", "PSRCC", "chromashift", "melspect", "specCont", "tonnetz"
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
    for person in people:
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
                '''
                # Layer two of feature selection using PCA
                if (len(X_train.columns) < feature_count):
                    feature_count = len(X_train.columns)
                    print("too little features pass correlation PCA used is: ", feature_count)
                pca = PCA(n_components=feature_count)
                pca.fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)

                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.fit_transform(X_test)

                Trainning_Set = (tf.data.Dataset.from_tensor_slices((tf.cast(X_train, tf.float32),
                                                                     tf.cast(Y_train, tf.int32))))

                # Test_Set = (tf.data.Dataset.from_tensor_slices((tf.cast(X_test, tf.float32),
                #                                                tf.cast(Y_test, tf.int32))))

                print(Trainning_Set)
                # print(Test_Set)
                if (str(item) == "chromashift"):
                    model = KerasClassifier(build_fn=create_model_Chromashift, verbose=0)
                elif (str(item) == "specCont"):
                    model = KerasClassifier(build_fn=create_model_specCont, verbose=0)
                elif (str(item) == "tonnetz"):
                    model = KerasClassifier(build_fn=create_model_tonnetz, verbose=0)
                else:
                    print(item == "chromashift")
                    model = KerasClassifier(build_fn=create_model, verbose=0)
                # grid search epochs, batch size and optimizer
                optimizers = ['adam']  # 'rmsprop',
                init = ['normal']  # 'glorot_uniform', 'uniform',
                epochs = [100]  # 50, 150
                batches = [25]  # 5, 10, 20, 50
                learn_rate = [.001]
                param_grid = dict(epochs=epochs, batch_size=batches, init=init,
                                  learn_rate=learn_rate)  # optimizer=optimizers, , momentum=momentum
                grid = GridSearchCV(estimator=model, param_grid=param_grid)
                grid_result = grid.fit(X_train, Y_train)
                # grid_results = grid_result.score(X_test, Y_test)
                # summarize results
                print("Training Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
                means = grid_result.cv_results_['mean_test_score']
                stds = grid_result.cv_results_['std_test_score']
                params = grid_result.cv_results_['params']
                for mean, stdev, param in zip(means, stds, params):
                    # print("Test scores for " + str(validUser) + " " + str(sound) + " %f (%f) with: %r" % (mean, stdev, param))
                    validation_info = [validUser, sound, "CNN (Adam)", means, stds, params]
                    validation_df = validation_df.append(pd.Series(validation_info, index=validation_df.columns),
                                                         ignore_index=True)

                pred_keras = grid_result.predict(X_test)
                matrix = confusion_matrix(Y_test.values.ravel(), pred_keras)
                TP, FN, FP, TN = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
                score_info = [validUser, sound, "CNN (Adam)", TP, FN, FP, TN, grid_result.get_params()]
                scores_df = scores_df.append(pd.Series(score_info, index=scores_df.columns), ignore_index=True)

                '''
                model_to_save = grid_result
                #best_model.save_model()
                converter = tf.lite.TFLiteConverter.from_keras_model(model_to_save)
                tflite_model = converter.convert()
                with tf.io.gfile.GFile("model.tflite", "wb") as f:
                    f.write(tflite_model)
                '''
    print("breakpoint save for ", item)
    writer = pd.ExcelWriter("Kera Results for PID" + " " + item + "_6_24.xlsx", engine='xlsxwriter')
    scores_df.to_excel(writer, sheet_name='Sheet1')
    validation_df.to_excel(writer, sheet_name="sheet2")
    writer.save()
    print(scores_df)