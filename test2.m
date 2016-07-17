% Show Prediction for Trained Network
clear all;
run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn;

path = '/home/qwertzuiopu/Dokumente/Uni/SS16/MLMI/Unet Project/';
load(fullfile(path, 'net-epoch-22.mat'));
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';

testpath = '/home/qwertzuiopu/data/datasets/val_set/';
testFiles = dir(fullfile(testpath,'T1','*.jpg'));
testFiles = struct2cell(testFiles)';
testFiles = testFiles(:,1);
testFilesout = dir(fullfile(testpath,'T2','*.jpg'));
testFilesout = struct2cell(testFilesout)';
testFilesout = testFilesout(:,1);
for f = 1 : size(testFiles,1)
    imdb.testFiles{f} = char(fullfile(testpath,'T1', testFiles(f)));
    imdb.testFilesout{f} = char(fullfile(testpath,'T2', testFilesout(f)));
end

a = 10;
errors = zeros(1,a*fix(size(testFiles,1)/a));
for i = 0 : fix(size(testFiles,1)/a)
    i
    images = imdb.testFiles((a*i+1):a*(i+1));
    imagesput = imdb.testFilesout((a*i+1):a*(i+1));
    input = vl_imreadjpeg(images, 'NumThreads', 4);
    input = cat(4, input{:});
    input = input(:,:,1,:);
    input = input / 255;
    input = single(input);
    inputout = vl_imreadjpeg(imagesput, 'NumThreads', 4);
    inputout = cat(4, inputout{:});
    inputout = inputout(:,:,1,:);
    inputout = inputout / 255;
    inputout = single(inputout);


    net.eval({'input',input});
    prediction = net.vars(net.getVarIndex('x45')).value;

    for j = 1 : a
        c = inputout(:,:,:,j);
        diff = prediction(:,:,:,j) - c;
        Y =  sqrt(sum( reshape(diff, [], size(c, 4)) .^ 2, 1)/(256*256));
        errors(1,(a*i+j)) = Y;
    end
end