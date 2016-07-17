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
images = imdb.testFiles(1:10)
images(9)
input = vl_imreadjpeg(images, 'NumThreads', 4);
input = cat(4, input{:});
input = input(:,:,1,:);
input = input / 255;
input = single(input);
    
net.eval({'input',input});
prediction = net.vars(net.getVarIndex('x45')).value;

imagesc(prediction(:,:,:,9));