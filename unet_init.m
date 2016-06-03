clear all
run /home/qwertzuiopu/.matconvnet-1.0-beta20/matlab/vl_setupnn

% Create DAGNN object
net = dagnn.DagNN();

%Create Building Blocks
poolBlock = dagnn.Pooling('poolSize', [2 2], 'stride', 2);

%Create Network Layers
net.addLayer('conv3x3_01', convBlock(3,3,1,64), {'x01'}, {'x02'}, {'f01', 'b01'});
net.addLayer('relu_01', dagnn.ReLU(), {'x02'}, {'x03'}, {});
net.addLayer('conv3x3_02', convBlock(3,3,64,64), {'x03'}, {'x04'}, {'f02', 'b02'});
net.addLayer('relu_02', dagnn.ReLU(), {'x04'}, {'x05'}, {});
net.addLayer('pool_01', poolBlock, {'x05'}, {'x06'}, {});

net.addLayer('conv3x3_03', convBlock(3,3,64,128), {'x06'}, {'x07'}, {'f03', 'b03'});
net.addLayer('relu_03', dagnn.ReLU(), {'x07'}, {'x08'}, {});
net.addLayer('conv3x3_04', convBlock(3,3,128,128), {'x08'}, {'x09'}, {'f04', 'b04'});
net.addLayer('relu_04', dagnn.ReLU(), {'x09'}, {'x10'}, {});
net.addLayer('pool_02', poolBlock, {'x10'}, {'x11'}, {});

net.addLayer('conv3x3_05', convBlock(3,3,128,256), {'x11'}, {'x12'}, {'f05', 'b05'});
net.addLayer('relu_05', dagnn.ReLU(), {'x12'}, {'x13'}, {});
net.addLayer('conv3x3_06', convBlock(3,3,256,256), {'x13'}, {'x14'}, {'f06', 'b06'});
net.addLayer('relu_06', dagnn.ReLU(), {'x14'}, {'x15'}, {});
net.addLayer('pool_03', poolBlock, {'x15'}, {'x16'}, {});

net.addLayer('conv3x3_07', convBlock(3,3,256,512), {'x16'}, {'x17'}, {'f07', 'b07'});
net.addLayer('relu_07', dagnn.ReLU(), {'x17'}, {'x18'}, {});
net.addLayer('conv3x3_08', convBlock(3,3,512,512), {'x18'}, {'x19'}, {'f08', 'b08'});
net.addLayer('relu_08', dagnn.ReLU(), {'x19'}, {'x20'}, {});
net.addLayer('pool_04', poolBlock, {'x20'}, {'x21'}, {});

net.addLayer('conv3x3_09', convBlock(3,3,512,1024), {'x21'}, {'x22'}, {'f09', 'b09'});
net.addLayer('relu_09', dagnn.ReLU(), {'x22'}, {'x23'}, {});
net.addLayer('conv3x3_10', convBlock(3,3,1024,1024), {'x23'}, {'x24'}, {'f10', 'b10'});
net.addLayer('relu_10', dagnn.ReLU(), {'x24'}, {'x25'}, {});
net.addLayer('upconv_01', convtBlock(2,2,1024,512), {'x25'}, {'x26'}, {'uf01', 'ub01'});

net.addLayer('crop_01', dagnn.Crop(), {'x20', 'x26'}, {'x27'}, {});
net.addLayer('concat_01', dagnn.Concat('dim', 3), {'x27', 'x26'}, {'x28'}, {});
net.addLayer('conv3x3_11', convBlock(3,3,1024,512), {'x28'}, {'x29'}, {'f11', 'b11'});
net.addLayer('relu_11', dagnn.ReLU(), {'x29'}, {'x30'}, {});
net.addLayer('conv3x3_12', convBlock(3,3,512,512), {'x30'}, {'x31'}, {'f12', 'b12'});
net.addLayer('relu_12', dagnn.ReLU(), {'x31'}, {'x32'}, {});
net.addLayer('upconv_02', convtBlock(2,2,512,256), {'x32'}, {'x33'}, {'uf02', 'ub02'});

net.addLayer('crop_02', dagnn.Crop(), {'x15', 'x33'}, {'x34'}, {});
net.addLayer('concat_02', dagnn.Concat('dim', 3), {'x34', 'x33'}, {'x35'}, {});
net.addLayer('conv3x3_13', convBlock(3,3,512,256), {'x35'}, {'x36'}, {'f13', 'b13'});
net.addLayer('relu_13', dagnn.ReLU(), {'x36'}, {'x37'}, {});
net.addLayer('conv3x3_14', convBlock(3,3,256,256), {'x37'}, {'x38'}, {'f14', 'b14'});
net.addLayer('relu_14', dagnn.ReLU(), {'x38'}, {'x39'}, {});
net.addLayer('upconv_03', convtBlock(2,2,256,128), {'x39'}, {'x40'}, {'uf03', 'ub03'});

net.addLayer('crop_03', dagnn.Crop(), {'x10', 'x40'}, {'x41'}, {});
net.addLayer('concat_03', dagnn.Concat('dim', 3), {'x41', 'x40'}, {'x42'}, {});
net.addLayer('conv3x3_15', convBlock(3,3,256,128), {'x42'}, {'x43'}, {'f15', 'b15'});
net.addLayer('relu_15', dagnn.ReLU(), {'x43'}, {'x44'}, {});
net.addLayer('conv3x3_16', convBlock(3,3,128,128), {'x44'}, {'x45'}, {'f16', 'b16'});
net.addLayer('relu_16', dagnn.ReLU(), {'x45'}, {'x46'}, {});
net.addLayer('upconv_04', convtBlock(2,2,128,64), {'x46'}, {'x47'}, {'uf04', 'ub04'});

net.addLayer('crop_04', dagnn.Crop(), {'x05', 'x47'}, {'x48'}, {});
net.addLayer('concat_04', dagnn.Concat('dim', 3), {'x48', 'x47'}, {'x49'}, {});
net.addLayer('conv3x3_17', convBlock(3,3,128,64), {'x49'}, {'x50'}, {'f17', 'b17'});
net.addLayer('relu_17', dagnn.ReLU(), {'x50'}, {'x51'}, {});
net.addLayer('conv3x3_18', convBlock(3,3,64,64), {'x51'}, {'x52'}, {'f18', 'b18'});
net.addLayer('relu_18', dagnn.ReLU(), {'x52'}, {'x53'}, {});
net.addLayer('conv1x1_01', convBlock(1,1,64,2), {'x53'}, {'x54'}, {'f19', 'b19'});


%Initialise random parameters
net.initParams();

%Visualize Network
net.print({'x01', [572 572 1]}, 'all', true, 'format', 'dot')

%Receptive Fields
%net.getVarReceptiveFields('x01').size