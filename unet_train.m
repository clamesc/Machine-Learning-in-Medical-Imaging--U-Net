function [net, info] = unet_train(varargin)

run  /Users/Leonard/Documents/MATLAB/TUM/MLMI/matconvnet/matlab/vl_setupnn

% define training data
opts.dataDir = fullfile('myfolder','mysubfolder','myfile.m');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% Prepare model and data
net = unet_init.m;       
imdb = load(opts.dataDir) ;

% Train the model
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;
