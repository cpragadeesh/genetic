function FeatIndex =  Genetic_Algorithm
   
   arg_list = argv();
   arg_list
   global num_data
   global text_data
   num_data = csvread('./pyout_num_data.csv');                       % gets all features into num_data and all cancer type or benign/malignant into text_data
   text_data = csvread('./pyout_text_data.csv')
   size(data)
   data(:, end)
   num_data = data(:,1:end-1);
   text_data = data(:,end);

   options = gaoptimset('CreationFcn', {@PopFunction},...
                        'PopulationSize',81,...                                            % Size of the population
                        'Generations',100,...                                              % Number of generations to run
                        'PopulationType', 'bitstring',...
                        'SelectionFcn',{@selectiontournament,2},...
                        'MutationFcn',{@mutationuniform, 0.1},...                          % Mutation probability
                        'CrossoverFcn', {@crossoverarithmetic,0.8},...                     % Crossover probability
                        'EliteCount',2,...                                                 % Number of elite children
                        'StallGenLimit',50,...
                        'PlotFcns',{@gaplotbestf},...  
                        'Display', 'iter');
    
   nVars = 779;                                                                              % Number of features in the dataset
   [chromosome] = ga(@FitFunc_KNN,nVars,options);                                          % calling the inbuilt genetic algorithm function with fitness function defined below, and options provided above.
   FeatIndex = find(chromosome==1);                                                        % Finds and returns the indices of elements having 1 in the final iteration of matrix
end
    
%% POPULATION FUNCTION
function [pop] = PopFunction(GenomeLength,~,options)
   pop = (rand(options.PopulationSize, GenomeLength)> rand);                               % Creates and returns a random binary matrix i.e. 0's and 1's
end
 
%% FITNESS FUNCTION  
function [FitVal] = FitFunc_KNN(pop)
   global num_data
   global text_data                                                                        % global variable to get num_data and text_data found above
   FeatIndex = find(pop==1);                                                               % finds indices of elements with value 1 from the random matrix found above
   if numel(FeatIndex)==0
       FeatIndex = ones(1,779);                                                              %Sometimes all the elements in the matrix may be 0's. In that case, the algorithm doesn't work, this function overcomes that case by setting all elements to 1
   end

   Y = grp2idx(text_data);                                                                 % it gives values for the last column. ex: if malignant and benign are the elements of the last column, then it gives 1 to benign/malignant whichever comes first and 2 to the other. i.e. it just groups and gives values from 1...
   X = num_data(:,[FeatIndex]);                                                            % gets the actual values of elements in database corresponding to the 1's found using FeatIndex
   NumFeat = numel(FeatIndex);                                                             % number of features selected i.e. number of 1's
   Compute = ClassificationKNN.fit(X,Y,'NSMethod','exhaustive','Distance','euclidean');    % fitness function. returns the fitness value
   Compute.NumNeighbors = 4;                                                               % number of nearest neighbours for KNN
   alpha = resubLoss(Compute);                                                             % Loss of k-nearest neighbor classifier by resubstitution
   %%%Fitness value 1
   FitVal = alpha/(779-NumFeat);
    
   %%%Fitness value 2
   FitVal = (alpha/NumFeat)+exp(-1/NumFeat);
end