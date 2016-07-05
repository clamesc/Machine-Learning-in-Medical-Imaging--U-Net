function net = unet_init()
    % Create DAGNN object
    net = dagnn.DagNN();

    %Create Building Blocks
    poolBlock = dagnn.Pooling('poolSize', [2 2], 'stride', 2);

    %Create Network Layers
    net.addLayer('conv3x3_01', convBlock(3,3,1,64), {'input'}, {'x02'}, {'f01', 'b01'});
    net.addLayer('relu_02', dagnn.ReLU(), {'x02'}, {'x03'}, {});
    net.addLayer('conv3x3_03', convBlock(3,3,64,64), {'x03'}, {'x04'}, {'f03', 'b03'});
    net.addLayer('relu_04', dagnn.ReLU(), {'x04'}, {'x05'}, {});
    net.addLayer('pool_05', poolBlock, {'x05'}, {'x06'}, {});
    
    net.addLayer('conv3x3_06', convBlock(3,3,64,128), {'x06'}, {'x07'}, {'f06', 'b06'});
    net.addLayer('relu_07', dagnn.ReLU(), {'x07'}, {'x08'}, {});
    net.addLayer('conv3x3_08', convBlock(3,3,128,128), {'x08'}, {'x09'}, {'f08', 'b08'});
    net.addLayer('relu_09', dagnn.ReLU(), {'x09'}, {'x10'}, {});
    net.addLayer('pool_10', poolBlock, {'x10'}, {'x11'}, {});
    
    net.addLayer('conv3x3_11', convBlock(3,3,128,256), {'x11'}, {'x12'}, {'f11', 'b11'});
    net.addLayer('relu_12', dagnn.ReLU(), {'x12'}, {'x13'}, {});
    net.addLayer('conv3x3_13', convBlock(3,3,256,256), {'x13'}, {'x14'}, {'f13', 'b13'});
    net.addLayer('relu_14', dagnn.ReLU(), {'x14'}, {'x15'}, {});
    net.addLayer('pool_15', poolBlock, {'x15'}, {'x16'}, {});
    
    net.addLayer('conv3x3_16', convBlock(3,3,256,512), {'x16'}, {'x17'}, {'f16', 'b16'});
    net.addLayer('relu_17', dagnn.ReLU(), {'x17'}, {'x18'}, {});
    net.addLayer('conv3x3_18', convBlock(3,3,512,512), {'x18'}, {'x19'}, {'f18', 'b18'});
    net.addLayer('relu_19', dagnn.ReLU(), {'x19'}, {'x20'}, {});
    net.addLayer('pool_20', poolBlock, {'x20'}, {'x21'}, {});
    
    net.addLayer('conv3x3_21', convBlock(3,3,512,1024), {'x21'}, {'x22'}, {'f21', 'b21'});
    net.addLayer('relu_22', dagnn.ReLU(), {'x22'}, {'x23'}, {});
    net.addLayer('conv3x3_23', convBlock(3,3,1024,1024), {'x23'}, {'x24'}, {'f23', 'b23'});
    net.addLayer('relu_24', dagnn.ReLU(), {'x24'}, {'x25'}, {});
    net.addLayer('upconv_25', convtBlock(2,2,512,1024), {'x25'}, {'x26'}, {'f25', 'b25'});
    
    %net.addLayer('crop_26', dagnn.Crop(), {'x20', 'x26'}, {'x27'}, {});
    %net.addLayer('concat_27', dagnn.Concat('dim', 3), {'x27', 'x26'}, {'x28'}, {});
    net.addLayer('conv3x3_28', convBlock(3,3,512,512), {'x26'}, {'x29'}, {'f28', 'b28'});
    net.addLayer('relu_29', dagnn.ReLU(), {'x29'}, {'x30'}, {});
    net.addLayer('conv3x3_30', convBlock(3,3,512,512), {'x30'}, {'x31'}, {'f30', 'b30'});
    net.addLayer('relu_31', dagnn.ReLU(), {'x31'}, {'x32'}, {});
    net.addLayer('upconv_32', convtBlock(2,2,256,512), {'x32'}, {'x33'}, {'f32', 'b32'});
    
    %net.addLayer('crop_33', dagnn.Crop(), {'x15', 'x33'}, {'x34'}, {});
    %net.addLayer('concat_34', dagnn.Concat('dim', 3), {'x34', 'x33'}, {'x35'}, {});
    net.addLayer('conv3x3_35', convBlock(3,3,256,256), {'x33'}, {'x36'}, {'f35', 'b35'});
    net.addLayer('relu_36', dagnn.ReLU(), {'x36'}, {'x37'}, {});
    net.addLayer('conv3x3_37', convBlock(3,3,256,256), {'x37'}, {'x38'}, {'f37', 'b37'});
    net.addLayer('relu_38', dagnn.ReLU(), {'x38'}, {'x39'}, {});
    net.addLayer('upconv_39', convtBlock(2,2,128,256), {'x39'}, {'x40'}, {'f39', 'b39'});
    
    %net.addLayer('crop_40', dagnn.Crop(), {'x10', 'x40'}, {'x41'}, {});
    %net.addLayer('concat_41', dagnn.Concat('dim', 3), {'x41', 'x40'}, {'x42'}, {});
    net.addLayer('conv3x3_42', convBlock(3,3,128,128), {'x40'}, {'x43'}, {'f42', 'b42'});
    net.addLayer('relu_43', dagnn.ReLU(), {'x43'}, {'x44'}, {});
    net.addLayer('conv3x3_44', convBlock(3,3,128,128), {'x44'}, {'x45'}, {'f44', 'b44'});
    net.addLayer('relu_45', dagnn.ReLU(), {'x45'}, {'x46'}, {});
    net.addLayer('upconv_46', convtBlock(2,2,64,128), {'x46'}, {'x47'}, {'f46', 'b46'});
    
    %net.addLayer('crop_47', dagnn.Crop(), {'x05', 'x47'}, {'x48'}, {});
    %net.addLayer('concat_48', dagnn.Concat('dim', 3), {'x48', 'x47'}, {'x49'}, {});
    net.addLayer('conv3x3_49', convBlock(3,3,64,64), {'x47'}, {'x50'}, {'f49', 'b49'});
    net.addLayer('relu_50', dagnn.ReLU(), {'x50'}, {'x51'}, {});
    net.addLayer('conv3x3_51', convBlock(3,3,64,64), {'x51'}, {'x52'}, {'f51', 'b51'});
    net.addLayer('relu_52', dagnn.ReLU(), {'x52'}, {'x53'}, {});
    net.addLayer('conv1x1_53', convBlock(1,1,64,1), {'x53'}, {'predictions'}, {'f53', 'b53'});
    
    net.addLayer('loss', dagnn.Loss('loss', 'regloss'), {'predictions', 'labels'}, 'objective') ;

    %Initialise random parameters
    net.initParams();
    
    %Visualize Network
    net.print({'input', [428 428 1]}, 'all', true, 'format', 'dot')
    
    %Receptive Fields
    %net.getVarReceptiveFields('x01').size
end