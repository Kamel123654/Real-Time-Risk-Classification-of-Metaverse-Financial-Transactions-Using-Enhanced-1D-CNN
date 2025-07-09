% Define the path to your Excel file
filename = 'D:\New folder (56)/metaverse_transactions_dataset.csv'; % Replace 'path_to_your_file' with the actual path

% Define the path to your Excel file

% Specify the sheet name
%sheetName = 'metaverse_transactions_dataset'; % As per your Excel file

% Set import options to preserve the original column headers
opts = detectImportOptions(filename,'VariableNamingRule', 'preserve');

% Read the dataset into a table
tbl = readtable(filename, opts);
 %% 2. Parse timestamp and extract features
    % Assuming 'timestamp' column exists and is in ISO format
    tbl.timestamp = datetime(tbl.timestamp, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    % Extract time-based features
    tbl.Hour       = hour(tbl.timestamp);
    tbl.DayOfWeek  = weekday(tbl.timestamp);   % 1=Sunday, ..., 7=Saturday
    tbl.Day        = day(tbl.timestamp);
    tbl.Month      = month(tbl.timestamp);
    tbl.Year       = year(tbl.timestamp);
labelName = "anomaly";
tbl = convertvars(tbl,labelName,'categorical');
%tbl(:, 1:2) = [];
categoricalInputNames = {'transaction_type', 'location_region', 'purchase_pattern', 'age_group', 'anomaly'};

tbl = convertvars(tbl,categoricalInputNames,'categorical');
for i = 1:numel(categoricalInputNames)
    name = categoricalInputNames(i);
    oh = onehotencode(tbl(:,name));
    tbl = addvars(tbl,oh,'After',name);
    tbl(:,name) = [];
end
tbl = splitvars(tbl);
% Load the Excel file
dataTable = tbl;
% Example to consolidate risk categories
riskCategories = {'high_risk', 'low_risk', 'moderate_risk'};
labelData = zeros(height(dataTable), 1); % Initialize label data

% Iteratively check each category; this assumes only one of these categories is 'true' per row
for idx = 1:length(riskCategories)
    % Find rows that are true for the current category
    rows = dataTable.(riskCategories{idx}) == 1;
    labelData(rows) = idx; % Assign a numeric label based on the risk
end

% Convert numeric labels to categorical
labels = categorical(labelData, 1:length(riskCategories), riskCategories);

% Remove the risk category columns from the dataTable
dataTable(:, riskCategories) = [];
% Existing code above that generates the 'labels' variablelabelData
% Now, add labels as a new column to dataTable
dataTable.labels = labelData;

targetCount = 63494; % The target count to match
tbl1 = dataTable;
% Categories to be oversampled
categories = [1, 3];

% Loop through each category that needs to be oversampled
for category = categories
    % Identify rows for the current category
    categoryRows = dataTable(dataTable.labels == category, :);
    % Calculate the current and additional rows needed
    currentRowsCount = height(categoryRows);
    additionalRowsNeeded = targetCount - currentRowsCount; % Total extra rows needed to reach targetCount
    % Replicate the category rows until we reach or exceed the target number of rows
    replicatedRows = repmat(categoryRows, ceil(additionalRowsNeeded / currentRowsCount), 1);
    % If we have more rows than needed, truncate the extra rows
    replicatedRows = replicatedRows(1:additionalRowsNeeded, :);
    % Combine original dataset with replicated rows
    dataTable = [dataTable; replicatedRows];
end
% Shuffle the rows to mix the replicated rows with the original dataset
dataTable = dataTable(randperm(height(dataTable)), :);
% Verify the operation performed correctly
disp(sum(dataTable.labels == 2)); % Expect this to be close to targetCount
disp(sum(dataTable.labels == 3)); % Expect this to be close to targetCount
% Now dataTable has the 'labels' column with the categorical data appended to it
% Assuming the remaining processing steps in your script are correct,
% now, you should have features in "dataTable" without the risk label columns.
% Your "labels" variable has the consolidated categories as a single categorical array.
% Assuming dataTable has been processed as per your code snippet, excluding label columns.
dataTable.timestamp = []; 
YY = categorical(dataTable.labels);
sFeat = table2array(dataTable);

kk = sFeat;
kk=kk';
kkk = kk;
%kkk = array2table(kk);
for i=1:size(kkk,2)
    kkkk{i,1} = kkk(:,i);
end
kk=kkkk;
hhh = YY;
numObservations = size(kk,1);
numObservationsTrain = floor(0.7*numObservations);
numObservationsValidation = floor(0.15*numObservations);
numObservationsTest = numObservations - numObservationsTrain - numObservationsValidation;
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);
idxTest = idx(numObservationsTrain+numObservationsValidation+1:end);
tblTrain = kk(idxTrain,:);
tblValidation = kk(idxValidation,:);
tblTest = kk(idxTest,:);
labelsTrain = hhh(idxTrain,:);
labelsValidation = hhh(idxValidation,:);
labelsTest = hhh(idxTest,:);
numFeatures = size(tblTrain{1},1);
numClasses = 3;
filterSize = 5;
numFilters = size(kk,1);
filterSize = 3;
numFilters = 32;
dropoutFactor = 0.005;
miniBatchSize = 16;
layers = createInnovativeCNNArchitecture(numFeatures, numClasses);% Split your data here if not done previously


% Adjust the training options
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',100, ...
    'InitialLearnRate',1e-3, ... % Adjust learning rate
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'L2Regularization',1e-4,... % Adjust L2 regularization
    'ValidationData',{tblValidation,labelsValidation}, ...
    'Shuffle','every-epoch',...
    'executionEnvironment','cpu',...
    'Plots','training-progress',...
    'LearnRateSchedule', 'piecewise', ...    
    'LearnRateDropFactor', 0.5, ... % More aggressive learning rate drop
    'LearnRateDropPeriod',5); % Drop the learning rate every 5 epochs

% Train the network with the new architecture
% ... (rest of the code)

% Assuming 'hhh' contains your categorical labels

% Convert labels to a cell array of categorical vectors
% Convert the entire labels array to categorical if it's not already



[net, info] = trainNetwork(tblTrain, labelsTrain, layers, options);
YPred = classify(net,tblTest,'MiniBatchSize',miniBatchSize,'ExecutionEnvironment','cpu');
YTest = labelsTest';
lgraph=net;
deepNetworkDesigner(lgraph);
figure;
plot(lgraph);
ax = gca;
ax.XTick = [];
ax.YTick = [];
title('Architecture of CNN');
analyzeNetwork(lgraph);