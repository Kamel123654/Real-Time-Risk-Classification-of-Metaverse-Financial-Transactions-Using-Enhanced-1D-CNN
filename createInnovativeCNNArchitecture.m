function layers = createInnovativeCNNArchitecture(numFeatures, numClasses)
    % Define the layers of the CNN
    inputLayer = sequenceInputLayer(numFeatures, 'Name', 'input');
    
    convBlock1 = [
        convolution1dLayer(3, 64, 'Padding','same', 'Name', 'conv1_1'),
        batchNormalizationLayer('Name', 'bn1_1'),
        reluLayer('Name', 'relu1_1'),
        convolution1dLayer(3, 64, 'Padding','same', 'Name', 'conv1_2'),
        batchNormalizationLayer('Name', 'bn1_2'),
        reluLayer('Name', 'relu1_2')];
    
    resBlock = [
        convolution1dLayer(3, 64, 'Padding','same', 'Stride',2, 'Name', 'res_conv1'),
        batchNormalizationLayer('Name', 'res_bn1'),
        reluLayer('Name', 'res_relu1'),
        convolution1dLayer(3, 64, 'Padding','same', 'Name', 'res_conv2'),
        batchNormalizationLayer('Name', 'res_bn2'),
        reluLayer('Name', 'res_relu2')];
    
    attLayer = [
        globalAveragePooling1dLayer('Name', 'gap'),
        fullyConnectedLayer(numFeatures, 'Name', 'fc_attention'),
        softmaxLayer('Name', 'softmax_attention')];
    
    outputLayers = [
        fullyConnectedLayer(numClasses, 'Name', 'fc_out'),
        softmaxLayer('Name', 'softmax'),
        classificationLayer('Name', 'classification')];
    
    % Create a layer graph from the input layer
    lgraph = layerGraph(inputLayer);
    
    % Add all other layers to the layer graph
    lgraph = addLayers(lgraph, convBlock1);
    lgraph = addLayers(lgraph, resBlock);
    lgraph = addLayers(lgraph, attLayer);
    lgraph = addLayers(lgraph, outputLayers);
    
    % Connect all layers sequentially as per the CNN architecture
    lgraph = connectLayers(lgraph, 'input', 'conv1_1');
    lgraph = connectLayers(lgraph, 'relu1_2', 'res_conv1');
    lgraph = connectLayers(lgraph, 'res_relu2', 'gap');
    % Ensure the output from the attention mechanism feeds into your fully connected layer correctly
    lgraph = connectLayers(lgraph, 'softmax_attention', 'fc_out');
    
    % Convert the layer graph to a Layer array
    layers = lgraph.Layers;
    
    % Make sure to use analyzeNetwork(lgraph) to inspect the graph visually if needed
end