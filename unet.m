function [net, info] = unet( varargin )

    clear all;
    run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn;
    %addpath('C:\libs\matconvnet\matconvnet\matlab');
    %vl_setupnn;

    trainOpts.expDir = fullfile(pwd,'data');
    trainOpts.val = [];
    trainOpts.train = [];
    trainOpts.batchSize = 10;
    trainOpts.numSubBatches = 1;
    trainOpts.numEpochs = 20;
    trainOpts.continue = false;
    trainOpts.gpus = 1;
    trainOpts.learningRate = 0.001;
    trainOpts.momentum = 0.9 ;
    %trainOpts.plotStatistics = false;
    trainOpts.derOutputs = {'objective', 1};
    trainOpts = vl_argparse(trainOpts, varargin);
    
    net = unet_init();
    
    define imdb object with file names of dataset
    
    trainOpts.train
    trainOpts.val
    
    [net, info] = cnn_train_dag(net, imdbTool, @getBatch, trainOpts) ;
                                
    %net.eval({'input',data(:,:,:,1)});
    %prediction = net.vars(net.getVarIndex('prediction')).value;
    %segmentation = net.vars(net.getVarIndex('prob')).value;
    %figure(2);
    %imagesc(prediction(:,:,1))
    %figure(3);
    %imagesc(segmentation(:,:,1))

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
                  size(inputData,3));
    input(pad+1:end-pad,pad+1:end-pad,:,:) = inputData;
    
    % Create array on GPU
    input = gpuArray(input);
    
    % Load batch output
    output = vl_imreadjpeg(imdb.outFilenames(batch), 'NumThreads', 6);
    
    % Convert structure format into array
    output = cat(4, output{:});
    
    % Crop images to output size
    outputsize = 244;
    crop = (size(output,1) - outputsize)/2
    output = output(crop+1:end-crop,crop+1:end-crop,:,:);
    
    % Create array on GPU
    output = gpuArray();
    
    inputs = {'input', input, ...
              'labels', output};
end