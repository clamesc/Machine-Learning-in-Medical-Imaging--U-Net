function [net, info] = unet( varargin )

    run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn;
    %addpath('C:\libs\matconvnet\matconvnet\matlab');
    %vl_setupnn;

    % Training parameter
    trainOpts.expDir = fullfile(pwd,'data');
    trainOpts.val = [];
    trainOpts.train = [];
    trainOpts.batchSize = 20;
    trainOpts.numSubBatches = 1;
    trainOpts.numEpochs = 25;
    trainOpts.continue = false;
    trainOpts.gpus = []; %1
    trainOpts.learningRate = [10e-6*ones(1,5), 10e-7*ones(1,5), 10e-8*ones(1,5), 10e-9*ones(1,5), 10e-10*ones(1,5)];
    trainOpts.momentum = 0.95 ;
    %trainOpts.plotStatistics = false;
    trainOpts.derOutputs = {'objective', 1};
    trainOpts = vl_argparse(trainOpts, varargin);
    
    % Initialise CNN
    net = unet_init();
    
    % Set different Learning Rate for Transposed Convolutions
    convtFactor = 1.0;
    convtLR = trainOpts.learningRate * convtFactor;
    net.layers(25).learningRate = [convtLR, convtLR];
    net.layers(32).learningRate = [convtLR, convtLR];
    net.layers(39).learningRate = [convtLR, convtLR];
    net.layers(46).learningRate = [convtLR, convtLR];
    
    % Get Filenames
    inPath = '/home/qwertzuiopu/data/2d_images_extr/T1/';
    outPath = '/home/qwertzuiopu/data/2d_images_extr/T2/';
    
    inFiles = dir(fullfile(inPath,'*.jpg'));
    outFiles = dir(fullfile(outPath,'*.jpg'));
    inFiles = struct2cell(inFiles)';
    outFiles = struct2cell(outFiles)';
    inFiles = inFiles(:,1);
    outFiles = outFiles(:,1);
    
    % Reduce Dataset for Testing
    testNumber = 1000;
    inFiles = inFiles(1:testNumber,1);
    outFiles = outFiles(1:testNumber,1);
    
    % Define Image Database
    for t = 1 : size(inFiles,1)
        imdb.inFilenames{t} = fullfile(inPath, char(inFiles(t,1)));
        imdb.outFilenames{t} = fullfile(outPath, char(outFiles(t,1)));
    end
    
    % Define Training and Validation Set
    trainRatio = 0.8;
    trainRatio = round(size(inFiles,1)*trainRatio);
    trainOpts.train = sort(randsample(size(inFiles,1), trainRatio));
    trainOpts.val = setdiff(1:size(inFiles,1),trainOpts.train);
    
    % Train Network
    [net, info] = cnn_train_dag(net, imdb, @getBatch, trainOpts) ;
    
    % Show Prediction for Trained Network
    inputData = vl_imreadjpeg(imdb.inFilenames(1), 'NumThreads', 6);
    inputData = cat(4, inputData{:});
    inputsize = 428;
    pad = (inputsize - size(inputData,1))/2;
    input = zeros(size(inputData,1)+2*pad, ...
                  size(inputData,2)+2*pad, ...
                  1, ...
                  size(inputData,4));
    input(pad+1:end-pad,pad+1:end-pad,:,:) = inputData;
    input = single(input);
    net.eval({'input',input});
    prediction = net.vars(net.getVarIndex('predictions')).value;
    imagesc(prediction)

end

function inputs = getBatch(imdb, batch, varargin)

    % Load batch input
    inputData = vl_imreadjpeg(imdb.inFilenames(batch), 'NumThreads', 6);
    
    % Convert structure format into array
    inputData = cat(4, inputData{:});
    
    % Add padding to images to match inputsize
    inputsize = 428;
    pad = (inputsize - size(inputData,1))/2;
    input = zeros(size(inputData,1)+2*pad, ...
                  size(inputData,2)+2*pad, ...
                  1, ...
                  size(inputData,4));
    input(pad+1:end-pad,pad+1:end-pad,:,:) = inputData;
    input = input / 255;
    input = single(input);
    
    % Create array on GPU
    %input = gpuArray(input);
    
    % Load batch output
    output = vl_imreadjpeg(imdb.outFilenames(batch), 'NumThreads', 6);
    
    % Convert structure format into array
    output = cat(4, output{:});
    
    % Crop images to output size
    outputsize = 244;
    crop = (size(output,1) - outputsize)/2;
    output = output(crop+1:end-crop,crop+1:end-crop,:,:);
    output = output / 255;
    output = single(output);
    
    %for i = 1:14
    %    figure(1);
    %    imagesc(input(:,:,:,i))
    %    figure(2);
    %    imagesc(output(:,:,:,i))
    %    pause;
    %end
    
    % Create array on GPU
    %output = gpuArray(output);
    
    inputs = {'input', input, ...
              'labels', output};
end