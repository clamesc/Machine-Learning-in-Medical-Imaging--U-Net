% Show Prediction for Trained Network

path = '/home/qwertzuiopu/Dokumente/Uni/SS16/MLMI/Unet Project/data/';
load(fullfile(path, 'net-epoch-21.mat'));
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';

testpath = '/home/qwertzuiopu/Dropbox/MLMI/Ersten 20 Bilder/train/T1/';
testFiles = dir(fullfile(testpath,'*.jpg'));
testFiles = struct2cell(testFiles)';
testFiles = testFiles(:,1);
for f = 1 : size(testFiles,1)
    imdb.testFiles{f} = char(fullfile(testpath, testFiles(f)));
end

input = vl_imreadjpeg(imdb.testFiles(3), 'NumThreads', 4);
input = cat(4, input{:});
input = input(:,:,1,:);
input = input / 255;
input = single(input);
    
net.eval({'input',input});
prediction = net.vars(net.getVarIndex('x45')).value
max(max(prediction))
imagesc(prediction)