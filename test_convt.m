% Show Prediction for Trained Network
clear all;
run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn;

path = '/home/qwertzuiopu/Dokumente/Uni/SS16/MLMI/Unet Project/data_convt/';
load(fullfile(path, 'net-epoch-100.mat'));
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';

testpath = '/home/qwertzuiopu/data/datasets/test_set/T1/';
testFiles = dir(fullfile(testpath,'*.jpg'));
testFiles = struct2cell(testFiles)';
testFiles = testFiles(:,1);
for f = 1 : size(testFiles,1)
    imdb.testFiles{f} = char(fullfile(testpath, testFiles(f)));
end

images = imdb.testFiles(1:10)
input = vl_imreadjpeg(images, 'NumThreads', 4);
input = cat(4, input{:});

% Normalize inputdata
inputsize = 340;
crop = (size(input,1) - inputsize)/2;
input = input(crop+1:end-crop,crop+1:end-crop,1,:);
input = input / 255;
input = single(input);
    
net.eval({'input',input});
prediction = net.vars(net.getVarIndex('x42')).value;
imagesc(prediction(:,:,:,1));