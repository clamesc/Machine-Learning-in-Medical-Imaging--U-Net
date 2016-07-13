function [net, info] = unet( varargin )

    run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn;
    %addpath('C:\libs\matconvnet\matconvnet\matlab');
    %vl_setupnn;

    % Training parameter
    trainOpts.expDir = fullfile(pwd,'data');
    trainOpts.val = [];
    trainOpts.train = [];
    trainOpts.batchSize = 15;
    trainOpts.numSubBatches = 1;
    trainOpts.continue = true;
    trainOpts.gpus = []; %1
    trainOpts.learningRate = 1e-7*ones(1,100);
    trainOpts.weightDecay = 0.01;
    trainOpts.momentum = 0.9 ;
    trainOpts.numEpochs = numel(trainOpts.learningRate);
    %trainOpts.plotStatistics = false;
    trainOpts.derOutputs = {'objective', 1};
    trainOpts = vl_argparse(trainOpts, varargin);   
    % Initialise CNN
    net = unet_init4;
    
    % Get Filenames
    path = '/home/qwertzuiopu/Dropbox/MLMI/Ersten 20 Bilder/';
    
    trainInFiles = dir(fullfile(path,'train','T1','*.jpg'));
    trainOutFiles = dir(fullfile(path,'train','T2','*.jpg'));
    valInFiles = dir(fullfile(path,'val','T1','*.jpg'));
    valOutFiles = dir(fullfile(path,'val','T2','*.jpg'));
    trainInFiles = struct2cell(trainInFiles)';
    trainOutFiles = struct2cell(trainOutFiles)';
    valInFiles = struct2cell(valInFiles)';
    valOutFiles = struct2cell(valOutFiles)';
    trainInFiles = trainInFiles(:,1);
    trainOutFiles = trainOutFiles(:,1);
    valInFiles = valInFiles(:,1);
    valOutFiles = valOutFiles(:,1);
    for f = 1 : size(trainInFiles,1)
        trainInFiles(f) = fullfile(path, 'train', 'T1', trainInFiles(f));
        trainOutFiles(f) = fullfile(path, 'train', 'T2', trainOutFiles(f));
    end
    for f = 1 : size(valInFiles,1)
        valInFiles(f) = fullfile(path, 'val', 'T1', valInFiles(f));
        valOutFiles(f) = fullfile(path, 'val', 'T2', valOutFiles(f));
    end
    inFiles = cat(1,trainInFiles,valInFiles);
    outFiles = cat(1,trainOutFiles,valOutFiles);
    
    % Define Training and Validation Set
    trainOpts.train = 1 : size(trainInFiles,1);
    trainOpts.val = (size(trainInFiles,1)+1) : (size(trainInFiles,1)+size(valInFiles,1));
    
    % Define Image Database
    for t = 1 : size(inFiles,1)
        imdb.inFilenames{t} = char(inFiles(t));
        imdb.outFilenames{t} = char(outFiles(t));
    end
    
    % Train Network
    [net, info] = cnn_train_dag(net, imdb, @getBatch, trainOpts) ;
end

function inputs = getBatch(imdb, batch, varargin)

    % Load batch input
    input = vl_imreadjpeg(imdb.inFilenames(batch), 'NumThreads', 4);
    
    % Convert structure format into array
    input = cat(4, input{:});
    
    % Normalize inputdata
    input = input(:,:,1,:) / 255;
    input = single(input);
    
    % Create array on GPU
    %input = gpuArray(input);
    
    % Load batch output
    output = vl_imreadjpeg(imdb.outFilenames(batch), 'NumThreads', 4);
    
    % Convert structure format into array
    output = cat(4, output{:});
    
    % Crop images to output size
    output = output(:,:,1,:) / 255;
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