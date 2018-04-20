%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% applying MATLAB neural net toolbox
function fitness = NN_Result(m,n,p,data)
% This function runs multiple iterations of NN with determined variables
% Input: m -> number of NN inputs
%        n -> a vector of number of neurons for each hidden layer
%        p -> number of NN outputs
%        data -> data points of the function for approximation
% Output: fitness -> returnes the value of objective function


% reading the data
inputs = data(1:m,:);
targets = data(m+1:m+p,:);

% create a fitting network
hiddenLayerSize = nonzeros(n)'; % takes nonzero elements of n as the number of neurons in hidden layers
net = fitnet(hiddenLayerSize); % define network
N_layer = length(hiddenLayerSize); % number of hidden layers
n_total = sum(hiddenLayerSize); % total number of hidden neurons

% set up division of data for training, validation, testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% training parameters
% net.trainParam.lr = 0.001; % uncomment for changing the learning rate
net.trainParam.max_fail=10;
net.trainParam.epochs = 100;
net.trainParam.showWindow = false;

% train the network 5 times and record the performance
J = zeros(1,5); % MSE of all trainings
epochs = zeros(1,5); % number of epochs of all trainings
for i = 1:5
    % train the network
    [ga_net,tr] = train(net,inputs,targets);
    % test the network
    outputs = ga_net(inputs);
    J(i) = perform(ga_net,targets,outputs); % MSE error
    epochs(i) = tr.num_epochs; % number of epochs
end
J_avg = mean(J); % MSE mean
J_range = range(J); % MSE range
N_epoch = mean(epochs); % average number of epochs

% compute fitness function
fitness = 0.001*(1000*J_avg + 0.5*n_total + N_layer +...
    100*J_range + 0.1/100*N_epoch);

end