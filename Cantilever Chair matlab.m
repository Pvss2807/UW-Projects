%REGRESSION MODEL

% Step 1: Load the training dataset
file_path = 'C:\Users\IdeasClinicCoops\Downloads\Training_Data.csv';
data_table = readtable(file_path);

% Step 2: Prepare the input and target data for training
input_data = table2array(data_table(:, 3:end));  % Extract features from the table
target_data = table2array(data_table(:, 1));     % Maximum Load Applied (in kg) is the target

% Step 3: Create the ANN model
hidden_layer_size = 10;  % You can adjust the number of hidden layers as needed
rng(123);  % This sets the seed of the random number generator
ann_model = feedforwardnet(hidden_layer_size);

% Step 4: Train the ANN model on the training data
ann_model = train(ann_model, input_data', target_data');

% Step 5: Load the test dataset
test_file_path = 'C:\Users\IdeasClinicCoops\Downloads\Test_Data.csv';
test_data_table = readtable(test_file_path);

% Step 6: Prepare the input data for testing
test_input_data = table2array(test_data_table(:, 3:end));

% Step 7: Test the ANN model on the test data
predicted_output = ann_model(test_input_data');

% Display the predicted outputs
disp(predicted_output);

% Step 8: Get the actual output data
actual_output = table2array(test_data_table(:, 1));

% Step 9: Calculate the errors
errors = predicted_output - actual_output';

% Step 10: Calculate MSE (Mean Squared Error)
mse_value = mean(errors.^2);
disp(['MSE: ', num2str(mse_value)])

% Step 11: Calculate RMSE (Root Mean Squared Error)
rmse_value = sqrt(mse_value);
disp(['RMSE: ', num2str(rmse_value)])

% Step 12: Calculate MAE (Mean Absolute Error)
mae_value = mean(abs(errors));
disp(['MAE: ', num2str(mae_value)])

% Step 13: Calculate R-squared (coefficient of determination)
ss_total = sum((actual_output - mean(actual_output)).^2);
ss_residual = sum(errors.^2);
rsquared_value = 1 - (ss_residual / ss_total);
disp(['R-squared: ', num2str(rsquared_value)])


%CLASSIFICATION MODEL

% Load the Training Dataset
trainData = readtable('C:\Users\IdeasClinicCoops\Downloads\Training_Data.csv');

% Encode Your Target Variable for Training
target_train = categorical(trainData.MaximumLoadApplied_inKg_);

% Define the Neural Network
inputSize = width(trainData) - 1; % Number of input features
numClasses = numel(categories(target_train)); % Number of classes
layers = [
    featureInputLayer(inputSize)
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize',128, ...
    'Plots','training-progress');

% Train the Network
net = trainNetwork(table2array(trainData(:,2:end)), target_train, layers, options);

% Load the Test Dataset
testData = readtable('C:\Users\IdeasClinicCoops\Downloads\Test_Data.csv');

% Encode Your Target Variable for Testing
target_test = categorical(testData.MaximumLoadApplied_inKg_);

% Test the Network
YPred = classify(net, table2array(testData(:,2:end)));












rng(123)

% Load your dataset
data = readtable('C:\Users\IdeasClinicCoops\Downloads\Training_Data.csv');

% Separate into classes
class1 = data(1:50,:);
class2 = data(51:100,:);
class3 = data(101:150,:);
class4 = data(151:200,:);

% Concatenate your data back into one table with the class labels
data = [class1; class2; class3; class4];
data.Class = [ones(height(class1), 1); 2*ones(height(class2), 1); 3*ones(height(class3), 1); 4*ones(height(class4), 1)];

% Set up your training and testing data
cv = cvpartition(height(data), 'Holdout', 0.2);
idx = cv.test;
dataTrain = data(~idx,:);
dataTest  = data(idx,:);

% Train the ANN model
% Assuming 'Maximum Load Applied' is your response variable
predictorNames = {'Load_Application_Point_x_mm_', 'Load_Application_Point_y_mm_', 'Backrest_Armrest_Junction_x_mm_', 'Backrest_Armrest_Junction_y_mm_', 'Armrest_x_mm_', 'Armrest_y_mm_'};
predictors = dataTrain{:, predictorNames};

% Convert response to categorical for classification
responseCategorical = categorical(dataTrain{:,'Class'});

% Prepare validation predictors and responses
validationPredictors = dataTest{:, predictorNames};
validationResponsesCategorical = categorical(dataTest{:,'Class'});

% Define the layers for classification
layers = [
    featureInputLayer(6)
    fullyConnectedLayer(10)
    reluLayer
    fullyConnectedLayer(4) % output size should match the number of classes
    softmaxLayer
    classificationLayer];

% Set the options
options = trainingOptions('adam', ...
    'MaxEpochs',100, ...
    'ValidationData',{validationPredictors, validationResponsesCategorical}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(predictors, responseCategorical, layers, options);

% Predict the class of new data
classPred = classify(net, dataTest{:,predictorNames});

% Calculate the accuracy
accuracy = sum(classPred == categorical(dataTest.Class)) / length(classPred);

% Display the confusion matrix
confusionchart(categorical(dataTest.Class), classPred);

