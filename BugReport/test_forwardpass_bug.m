function [net, info] = unet( varargin )

    run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn;
    %addpath('C:\libs\matconvnet\matconvnet\matlab');
    %vl_setupnn;
    
    % Create DAGNN object
    net = dagnn.DagNN();
    net.addLayer('conv3x3_1', dagnn.Conv('size', [3 3 1 64], ...
        'hasBias', true), {'input'}, {'x2'}, {'f1', 'b1'});
    net.addLayer('relu_2', dagnn.ReLU(), {'x2'}, {'x3'}, {});
    net.addLayer('pool_3', dagnn.Pooling('poolSize', [2 2], ...
        'stride', [2 2]), {'x3'}, {'x4'}, {});
    net.addLayer('conv3x3_4', dagnn.Conv('size', [3 3 64 64], ...
        'hasBias', true), {'x4'}, {'x5'}, {'f4', 'b4'});
    net.addLayer('relu_38', dagnn.ReLU(), {'x5'}, {'x6'}, {});
    net.addLayer('conv1x1_41', dagnn.Conv('size', [1,1,64,1], ...
        'hasBias', true), {'x6'}, {'output'}, {'f41', 'b41'});
    net.initParams();
    net.mode = 'test';

    imdb = get_images();
    %--------------------------------------------------------------------
    % Forward Pass Bug depends on the number of images that are evaluated
    %--------------------------------------------------------------------
    images = imdb.testFiles(:);      % Use all 2 images for forward pass
    %images = imdb.testFiles(1);     % Use only 1 image for forward pass
    %--------------------------------------------------------------------
    
    
    % Read Images for Testing
    input = vl_imreadjpeg(images, 'NumThreads', 4);
    input = cat(4, input{:});
    input = input(:,:,1,:);
    input = input / 255;
    input = single(input);

    % Forward Pass
    net.eval({'input',input});
    prediction = net.vars(net.getVarIndex('output')).value;
    imagesc(prediction(:,:,:,1));
end

function imdb_obj = get_images()
    testFiles = dir(fullfile(pwd,'Images','*.jpg'));
    testFiles = struct2cell(testFiles)';
    testFiles = testFiles(:,1);
    for f = 1 : size(testFiles,1)
        imdb_obj.testFiles{f} = char(fullfile(pwd, ...
            'Images', testFiles(f)));
    end
end