function [net, info] = unet_train(varargin)

run  %/Users/.../vl_setupnn

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
[net, info] = trainfn(net, imdb, ...
  net.meta.trainOpts, ...
  opts.train);
  %'val', find(imdb.images.set == 3)) ;
end