%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code by Mahsa Ghasemi
% Fall 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate data as a function to be approximated by neural net

m = 1; % number of NN inputs
p = 1; % number of NN outputs
Np = 101; % number of data points
noise = 0.01; % noise level

x = linspace(-2, 2, Np); % input points
data_wo = zeros(2,Np); % data without noise
for i=1 : Np
    data_wo(:,i) = [x(i),sinc(3*x(i))];
end
data = data_wo + noise*randn(2,Np); % noisy data

%% genetic algorithm parameters
nvars = 3; % number of optimization variables
n_max = 10; % maximum number of neurons per hidden layer
LB = zeros(1,nvars); % lower bound of variables
UB = n_max*ones(1,nvars); % upper bound of variables
IntCon = [1:nvars]; % integer-valued variables
options = gaoptimset(...
    'PopulationType', 'doubleVector',...
    'PopulationSize', 25,...
    'PopInitRange', [zeros(1,4); n_max*ones(1,4)],...
    'EliteCount', 4,...
    'Generations', 50,...
    'TolFun', 1e-5,...
    'StallGenLimit', 25,...
    'Display', 'iter',...
    'PlotFcn',{@gaplotbestf, @gaplotbestindiv, @gaplotscorediversity,...
    @gaplotscores, @gaplotrange, @gaplotdistance, @gaplotstopping});

%% run genetic algorithm
h_fit = @(n) NN_Result(m,n,p,data); % handle to fitness function
[n_GA,fval,exitflag,output,population,scores] = ...
    ga(h_fit,nvars,[],[],[],[],LB,UB,[],IntCon,options); % run GA