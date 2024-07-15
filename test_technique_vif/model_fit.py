import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from keras.layers import GRU, LSTM, Dense
from keras.models import Sequential
from mlflow.models import infer_signature
from scipy.signal import resample
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")

if MLFLOW_TRACKING_URI is None:
    load_dotenv('dagshub.env')

@contextmanager
def timer(title: str):
    """
    A context manager that measures the time taken to execute a block of code.

    Args:
        title (str): The title of the code block.

    Yields:
        None

    Prints:
        The title of the code block and the time taken to execute it.

    Example:
        >>> with timer("Function execution"):
        ...     # code block
        Function execution - done in 5s
    """
    start_time = time.time()
    yield
    print(f"{title} - done in {time.time() - start_time:.0f}s")

def load_data(
        resampling: str='down',
        shuffle: bool=True,
        train_size: int=2000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data
    Function to load the data, resample it according to given strategy and
    eventually shuffle rows to avoid bias.
    Using PS2.txt and FS1.txt as features and profile.txt as target :
    - PS2.txt : 100 Hz
    - FS1.txt : 10 Hz
    - profile.txt : second column
    
    Parameters
    ----------
    resampling : Literal['up', 'down']
        Resampling method
    shuffle : bool
        Shuffle data
    train_size : int
        Train size
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, target_train, X_test, target_test
    """
    
    fs1_path = Path("data/FS1.txt")
    fs1_data_up = []
    
    with timer("loading FS1.txt"):
        with open(fs1_path) as f:
            x = np.arange(0, 6000, 10)
            for line in f:
                y = np.array([float(x) for x in line.split()])
                fs1_data_up.append(np.interp(np.arange(0, 6000), x, y))
                size_down = len(y)

        with open(fs1_path) as f:
            fs1_data = np.array(
                [float(v) for v in f.read().split()]
            ).reshape(-1, size_down)

    with timer("loading PS2.txt"):
        ps2_path = Path("data/PS2.txt")
        
        with open(ps2_path) as f:
            ps2_data = np.array(
                [float(v) for v in f.read().split()]
            ).reshape(-1, 6000)
            ps2_data_down = resample(ps2_data, size_down, axis=-1)

    with timer("loading profile.txt"):
        target = pd.read_csv("data/profile.txt", sep="\t", header=None)[1].values
    
    with timer("resampling data"):
        if resampling == 'up':
            data = np.stack([np.stack(fs1_data_up), ps2_data], axis=-1)
        elif resampling == 'down':
            data = np.stack([fs1_data, ps2_data_down], axis=-1)
    
    if shuffle:
        order = np.arange(len(data))
        np.random.shuffle(order)
        data = data[order]
        target = target[order]

    return (
        data[:train_size], target[:train_size],
        data[train_size:], target[train_size:]
    )

def train_model(
        X_train: np.ndarray,
        target_train: np.ndarray,
        X_test: np.ndarray,
        target_test: np.ndarray,
        layer: str,
        normalize: bool=False,
        resampling: str="down",
        shuffle: bool=True,
        train_size: int=2000
    ) -> None:
    """
    Train model
    Function to train the model.
    
    Parameters
    ----------
    X_train : np.ndarray
        Train features
    y_train : np.ndarray
        Train target
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test target
    normalize : bool
        Normalize data or not
    independent : bool
        Scale features independently or not.
    """

    if normalize:
        with timer("scaling data"):
            scaler = StandardScaler()
            # Reshape X_train_up to merge 'rows' and 'steps' dimensions
            X_train_scaled = scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[2])
            )
            # Reshape X_train_scaled_up back to the original shape
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_train = X_train_scaled

    with timer("encoding target"):
        encoder = LabelEncoder()
        encoder.fit(target_train)
        y_train = encoder.transform(target_train)
        y_test = encoder.transform(target_test)

    with timer("building model"):
        model = Sequential(name="Sequential")
        if layer == "gru":
            model.add(GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])))
        elif layer == "lstm":
            model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(4, activation='softmax'))

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    # Retrieving info for file annotations
    data_shuffle = "shuffled" if shuffle else "not_shuffled"
    data_scaling = type(scaler).__name__ if normalize else "no_scaler"
    artifact_path = (
        f"Model_{model.name}__Layer_{layer}__Resampling_{resampling}__"
        f"Scaling_{data_scaling}__Shuffling_{data_shuffle}"
    )
    model_name = f"{model.name}_{layer}"
    params = {
        'layer': layer,
        'resampling': resampling,
        'shuffle': shuffle,
        'train_size': train_size,
        'normalize': normalize
    }

    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    mlflow.set_experiment(experiment_name=artifact_path)
    try:
        _ = client.get_registered_model(name=model_name)
    except mlflow.exceptions.MlflowException:
        # If the model does not exist, use a default version number of "v1" for the run name
        run_name = f"{model_name}_v1"
    else:
        # If the model does exist, retrieve its latest version number and use that to construct the run name
        latest_version = client.get_latest_versions(name=model_name)[0].version
        version_number = int(latest_version.split("v")[-1]) + 1
        run_name = f"{model_name}_v{version_number}"

    with timer("mlflow run"):
        mlflow.start_run(run_name=run_name)

        with timer("fitting model"):
            model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1)

        with timer("saving model"):
            model.save(
                f"{artifact_path}.keras"
            )

        with timer("evaluating model"):
            scores_train = model.evaluate(X_train, y_train)
            scores_test = model.evaluate(X_test, y_test)
            metrics = {
                'train_loss': scores_train[0],
                'train_accuracy': scores_train[1],
                'test_loss': scores_test[0],
                'test_accuracy': scores_test[1]
            }

        with timer("logging params, metrics and model with mlflow"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.keras.log_model(
                model, artifact_path=artifact_path,
                registered_model_name=model_name,
                signature=signature
            )

        with timer("plotting confusion matrix"):
            y_predict = np.argmax(model.predict(X_train), axis=1)
            z = confusion_matrix(y_train, y_predict)
            z = (z/z.sum(axis=1)[:,None]).round(2)

            text_classes = [str(v) for v in encoder.classes_]
            fig = px.imshow(
                z, x=text_classes, y=text_classes, text_auto=True,
                color_continuous_scale=[
                    'MidnightBlue', 'Steelblue', 'Darkturquoise', 'Paleturquoise',
                    'Coral', 'Firebrick', 'Maroon'
                ]
            )

            fig.update_layout(
                title=f'Matrice de confusion - Modèle: {model.name}'
                    f' - Layer: {layer}<br>Resampling: {resampling} - '
                    f'Scaling: {data_scaling} - Shuffling: {data_shuffle}',
                xaxis_title="Predicted value",
                yaxis_title="Real value",
                title_x=0.5
            )

            with timer("saving confusion matrix"):
                image_name = (
                    f'Matrice_de_confusion__Model_{model.name}__'
                    f'Layer_{layer}__Resampling_{resampling}__'
                    f'Scaling_{data_scaling}__Shuffling_{data_shuffle}.png'
                )

                fig.write_image(image_name)

            with timer("logging confusion matrix"):
                mlflow.log_figure(fig, image_name)

        mlflow.end_run()

def main(
        layer: Literal["gru", "lstm"]="gru",
        resampling: str="down",
        shuffle: bool=True,
        train_size: int=2000,
        normalize: bool=False
    ) -> None:

    with timer("loading data"):
        X_train, target_train, X_test, target_test = load_data(
            resampling=resampling, shuffle=shuffle, train_size=train_size
        )
    with timer("training model"):
        train_model(
            X_train, target_train, X_test, target_test, layer=layer,
            normalize=normalize, resampling=resampling, shuffle=shuffle,
            train_size=train_size
        )

parameters_sets = [
    {
        'layer': "gru",
        'resampling': 'up',
        'shuffle': True,
        'train_size': 2000,
        'normalize': False
    }
]

for params in parameters_sets:
    main(**params)