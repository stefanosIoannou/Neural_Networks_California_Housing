import torch
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import make_column_transformer, ColumnTransformer, make_column_selector
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset


class CDataset(Dataset):
    """
    Class to create a dataset for the neural network
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.flatten(), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NN(nn.Module):
    """
    Neural network class
    """
    def __init__(self, input_size, output_size):
        """
        Initialize the neural network.

        Args:
            input_size: size of the input layer
            output_size: size of the output layer
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.Tanh(),
            nn.Linear(100, 70),
            nn.ReLU(),
            nn.Linear(70, 20),
            nn.ReLU(),
            nn.Linear(20, output_size)
        )

    def forward(self, x):
        """
        Forward pass of the neural network.
        Args:
            x: input data
        Returns:
            Output of the neural network
        """
        return self.layers(x).flatten()




class Regressor():

    def __init__(self, x, nb_epoch=1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocessing parameters
        self.one_hot_imputer_transformer = None
        self.one_hot_encoding_transformer = None
        self.normalizer = None

        # Preprocess data
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1

        # Model parameters
        self.NN = NN(self.input_size, self.output_size)

        # For testing Dropout
        # self.NN = DropoutNN(self.input_size, self.output_size)

        self.learning_rate = 0.00073378
        self.weight_decay = 0.0010347
        self.optimizer = torch.optim.Adam(self.NN.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.loss_function = nn.MSELoss()
        self.batch_size = 16
        self.nb_epoch = nb_epoch

        self._DEBUG = False
        self._TUNING = False

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if training:
            # Imputation of values and one-hot encoding
            self.one_hot_imputer_transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(),
                                                                   ['ocean_proximity']),
                                                                  ('imputer',
                                                                   SimpleImputer(
                                                                       missing_values=np.nan,
                                                                       strategy='mean'),
                                                                   make_column_selector(
                                                                       dtype_include=np.number))],
                                                                 remainder='passthrough')
            X_one_hot_imputed = pd.DataFrame(self.one_hot_imputer_transformer.fit_transform(x),
                                             columns=self.one_hot_imputer_transformer.get_feature_names_out())

            # Isolate categorical columns
            non_categorical_columns = []
            for category in X_one_hot_imputed.columns:
                if 'one_hot_encoder' not in category:
                    non_categorical_columns.append(category)

            # Standardization of numerical columns
            self.normalizer = make_column_transformer((StandardScaler(), non_categorical_columns),
                                                      remainder='passthrough')
            X_data_preprocessed = pd.DataFrame(self.normalizer.fit_transform(X_one_hot_imputed),
                                               columns=self.normalizer.get_feature_names_out())
        else:
            # Imputation of values and one-hot encoding from training data
            X_one_hot_imputed = pd.DataFrame(self.one_hot_imputer_transformer.transform(x),
                                             columns=self.one_hot_imputer_transformer.get_feature_names_out())
            X_data_preprocessed = pd.DataFrame(self.normalizer.transform(X_one_hot_imputed),
                                               columns=self.normalizer.get_feature_names_out())

        return X_data_preprocessed.to_numpy(), (y.to_numpy() if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train_loop(self, dataloader):
        """
        Train the model for one epoch.
        Args:
            dataloader: dataloader for the training data

        Returns:
            loss: loss of the model
        """
        training_loss = 0
        if not self._TUNING:
            self.NN.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.NN(X)
            loss = self.loss_function(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            training_loss += loss.item()
        return {'Training Loss': training_loss / len(dataloader)}

    def test_loop(self, dataloader):
        """
        Test the model for one epoch.
        Args:
            dataloader: dataloader for the test data

        Returns:
            loss: loss, R2 and MSE of the model
        """
        size = len(dataloader)
        test_loss = 0
        r2 = 0
        mse = 0
        if not self._TUNING:
            self.NN.eval()
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.NN(X)
                test_loss += self.loss_function(pred, y).item()
                r2 += r2_score(pred, y)
                mse = mean_squared_error(pred, y)

        return {'Test Loss': test_loss / size,
                'R2': r2 / size,
                'MSE': mse / size}

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocess data & convert to tensors
        X_train, y_train = self._preprocessor(x, y, training=True)
        train_dataset = CDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # For training + validaiton
        # train_dataset, val_dataset = self.preprocess_to_tensors(x, y)
        # val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # test_losses = []
        # r2 = []
        # mse = []

        # Evaluation parameters
        training_losses = []


        # Training loop
        for epoch in range(self.nb_epoch):
            training_res = self.train_loop(train_loader)
            # test_res = self.test_loop(val_loader)

            if self._DEBUG:
                print(f'Epoch: {epoch + 1}, {str(training_res)}')
                # print(f'Epoch: {epoch + 1}, {str(training_res)}, {str(test_res)}')

            training_losses.append(training_res['Training Loss'])
            # test_losses.append(test_res['Test Loss'])
            # mse.append(test_res['MSE'])
            # r2.append(test_res['R2'])

        if self._DEBUG:
            print("Done!")

        ## Print the learning curve when validating
        # self.plot_learning_curve(training_losses, test_losses)
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def plot_learning_curve(training_losses, test_losses):
        """
        Plot the loss curves for the training and test data sets.
        Args:
            training_losses: list of training losses
            test_losses: list of test losses
        """
        plt.plot(np.sqrt(training_losses), label='Train Loss')
        plt.plot(np.sqrt(test_losses), label='Test Loss')
        plt.legend()
        plt.show()

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################


        # Preprocess data
        X, _ = self._preprocessor(x, training=False)

        # Convert to torch tensors
        X = torch.tensor(X, dtype=torch.float32)

        # Forward pass
        y_pred = self.NN(X)

        return y_pred.detach().numpy()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X_test, y_test = self._preprocessor(x, y=y, training=False)
        test_dataset = CDataset(X_test, y_test)

        mseloss = nn.MSELoss()

        x, y = test_dataset[:]
        with torch.no_grad():
            self.NN.eval()
            y_predict = self.NN(x)
            mse = mseloss(y_predict, y).item()
            r2 = r2_score(y_predict, y)


        if not self._DEBUG:
            print(f'MSE: {mse}')
            print(f'R2 Score: {r2}')
            print(f'RMSE: {np.sqrt(mse)}')

        # return [mse, r2, np.sqrt(mse)]
        return np.sqrt(mse)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def preprocess_to_tensors(self, x, y):
        """
        Preprocess the data and convert it to torch tensors.
        Args:
            x: train and validation data inputs
            y: train and validation data outputs
        Returns:
            train_dataset: training and validation datasets
        """
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(x, y, random_state=4,
                                                          test_size=0.1,
                                                          shuffle=True)

        # Preprocess data
        X_train, y_train = self._preprocessor(X_train, y_train, training=True)
        X_val, y_val = self._preprocessor(X_val, y_val, training=False)

        # Convert to tensors
        train_dataset = CDataset(X_train, y_train)
        val_dataset = CDataset(X_val, y_val)

        return train_dataset, val_dataset


def split_data(data, seed=4):
    """
    Split the data into train, validation and test sets.

    Args:
        data: pandas dataframe containing the data
        seed: random seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test: train, validation and test sets
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(data.drop('median_house_value', axis=1),
                                                                data.median_house_value,
                                                                random_state=seed,
                                                                test_size=0.3,
                                                                shuffle=True)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                                                      random_state=seed,
                                                      test_size=0.1,
                                                      shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def subset_of_indices(train_dataset, val_dataset, train_perc=0.7, val_perc=0.4):
    """
    Return a subset of indices, randomly sampled from the train and validation
    dataset. Train_perc and Val_perc determine the size of the sample between (0,1)
    """
    train_indices = np.random.choice(np.array(range(len(train_dataset))),
                                     size=int(len(train_dataset) * train_perc),
                                     replace=False)
    val_indices = np.random.choice(np.array(range(len(val_dataset))),
                                   size=int(len(val_dataset) * val_perc),
                                   replace=False)
    return train_indices, val_indices


def train_and_validate(config):
    """
    Use ray to train and validate the model and print
    Args:
        config:

    Returns:

    """
    from ray import tune

    regressor = config['regressor']

    train_dataset = config["train_dataset"]
    val_dataset = config["val_dataset"]

    train_indices, val_indices = subset_of_indices(train_dataset, val_dataset)
    train_loader = DataLoader(Subset(train_dataset, train_indices),
                              batch_size=16,
                              shuffle=True)
    valid_loader = DataLoader(Subset(val_dataset, val_indices),
                              batch_size=len(val_dataset))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize the model.
    model = regressor.NN.to(device)
    regressor.loss_fn = nn.MSELoss()
    regressor.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    # Start the training.
    epochs = 100
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = regressor.train_loop(
            train_loader
        )
        val_metrics = regressor.test_loop(valid_loader)

        print(f"Training loss: {train_epoch_loss:.3f}")
        print(f"Validation loss: {val_metrics['Test Loss']:.3f}")
        print('-' * 50)

        tune.report(
            train_loss=train_epoch_loss,
            val_loss=val_metrics['Test Loss'],
            val_rmse=val_metrics['RMSE']
        )


def RegressorHyperParameterSearch(x, y):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        x training and validation data output
        y training and validation data input
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # Reference to: https://debuggercafe.com/hyperparameter-tuning-with-pytorch-and-ray-tune/

    from ray import tune

    regressor = Regressor(x, nb_epoch=100)
    train_dataset, val_dataset = regressor.preprocess_to_tensors(x, y)

    # Learning R. Weight Decay, Batch Size
    config = {
        "lr": tune.loguniform(6e-4, 9e-4),
        "weight_decay": tune.loguniform(1e-3, 2e-3),
        "batch_size": 16,
        "regressor": regressor,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset
    }

    # Schedule different instances of NNs to run
    scheduler = tune.schedulers.ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=80,
        grace_period=40,
        reduction_factor=2
    )

    # Report the results
    reporter = tune.CLIReporter(
        metric_columns=["train_loss", "val_loss"]
    )

    # Run and tune scheduled NNs
    result = tune.run(
        train_and_validate,
        resources_per_trial={"cpu": 4, "gpu": 0},
        config=config,
        num_samples=40,
        scheduler=scheduler,
        local_dir='raytune_results_mseloss',
        progress_reporter=reporter
    )

    # Other hyperparameter tuning/testing was done in the notebook or locally

    # to visualise hyperparameters use: tensorboard --logdir=./raytune_results_mseloss
    # following visual comparison of the training and validation loss,
    # the following hyperparameters have been selected
    return dict(learning_rate=0.00073378, weight_decay=0.0010347)

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():

    # This is just the example main that was provided.
    # The model was tuned and evaluated in the nn_pytorch.ipynb file

    output_label = "median_house_value"
    data = pd.read_csv("housing.csv")

    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    regressor = Regressor(x_train, nb_epoch=100)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error on training set
    error = regressor.score(x_train, y_train)
    print("\nRegressor error (training loss) (RMSE): {}\n".format(error))


if __name__ == "__main__":
    example_main()
