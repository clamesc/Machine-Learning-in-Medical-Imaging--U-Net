function Train50Salads( varargin )
    addpath('C:\libs\matconvnet\matconvnet\matlab');
    vl_setupnn;

    % -------------------------------------------------------------------------
    % CNN parameters 
    % -------------------------------------------------------------------------
    trainOpts.val = [];
    trainOpts.batchSize = 10;
    trainOpts.numSubBatches = 1;
    trainOpts.numEpochs = 20;
    trainOpts.continue = false;
    trainOpts.gpus = 1;
    trainOpts.learningRate = 0.001; %for l1 big: 0.00005;
    trainOpts.momentum = 0.9 ;
    %trainOpts.plotStatistics = false;    
    
    trainOpts.derOutputs = {'objective', 1};
    trainOpts = vl_argparse(trainOpts, varargin);

    netOpts.inNode = 5;
    netOpts.outNode = 1;
    netOpts.fc1rows = 5;
    netOpts.fc1cols = 7;
    
    for tool = 1:10
        fprintf('Loading imdb...\n');
        load(sprintf('imdb_50Salads_tool%i.mat',tool));
        imdbTool.labels = imdbTool.labels * 2 - 1;
        %imdb.labels = bsxfun(@rdivide, imdb.labels, sum(imdb.labels)/ mean(imdb.labels,1)); 

        fprintf('Training network...\n');
        tic
        trainOpts.expDir = fullfile(sprintf('network_flow_tool%i_train11', tool));
        trainOpts.train = find(imdbTool.videoId <= 151);
        trainOpts.val = find(imdbTool.videoId > 151);

        net = initializeNetwork(netOpts);
        net = cnn_init_dag(net);
        net.mode = 'normal';
        
        cnn_train_dag(net, imdbTool, @getBatch, trainOpts) ;
        t = toc;
        fprintf('\n Network trained. Time elapsed: %d sec \n', t);
    end
end

function inputs = getBatch(imdb, batch, varargin)
    input = vl_imreadjpeg(imdb.inFilenames(batch), 'NumThreads', 6);
    input = gpuArray(cat(4, input{:}));
    
    output = vl_imreadjpeg(imdb.outFilenames(batch), 'NumThreads', 6);
    output = gpuArray(cat(4, output{:}));
    
    inputs = {'input', input, ...
              'labels', output};
end