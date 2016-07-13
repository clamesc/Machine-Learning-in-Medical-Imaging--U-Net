function net = unet_init4()
    % Create DAGNN object
    net = dagnn.DagNN();

    %Create Network Layers
    net.addLayer('conv3x3_01', convBlock(3,3,1,64), {'input'}, {'x02'}, {'f01', 'b01'});
    net.addLayer('relu_02', dagnn.ReLU(), {'x02'}, {'x03'}, {});
    net.addLayer('conv3x3_03', convBlock(3,3,64,64), {'x03'}, {'x04'}, {'f03', 'b03'});
    net.addLayer('relu_04', dagnn.ReLU(), {'x04'}, {'x05'}, {});
    net.addLayer('pool_05', dagnn.Pooling('poolSize', [2 2], 'stride', 2), {'x05'}, {'x06'}, {});
    
    net.addLayer('conv3x3_06', convBlock(3,3,64,128), {'x06'}, {'x07'}, {'f06', 'b06'});
    net.addLayer('relu_07', dagnn.ReLU(), {'x07'}, {'x08'}, {});
    net.addLayer('conv3x3_08', convBlock(3,3,128,128), {'x08'}, {'x09'}, {'f08', 'b08'});
    net.addLayer('relu_09', dagnn.ReLU(), {'x09'}, {'x10'}, {});
    net.addLayer('pool_10', dagnn.Pooling('poolSize', [2 2], 'stride', 2), {'x10'}, {'x11'}, {});
    
    net.addLayer('conv3x3_11', convBlock(3,3,128,256), {'x11'}, {'x12'}, {'f11', 'b11'});
    net.addLayer('relu_12', dagnn.ReLU(), {'x12'}, {'x13'}, {});
    net.addLayer('conv3x3_13', convBlock(3,3,256,256), {'x13'}, {'x14'}, {'f13', 'b13'});
    net.addLayer('relu_14', dagnn.ReLU(), {'x14'}, {'x15'}, {});
    net.addLayer('pool_15', dagnn.Pooling('poolSize', [2 2], 'stride', 2), {'x15'}, {'x16'}, {});
    
    net.addLayer('conv3x3_16', convBlock(3,3,256,512), {'x16'}, {'x17'}, {'f16', 'b16'});
    net.addLayer('relu_17', dagnn.ReLU(), {'x17'}, {'x18'}, {});
    net.addLayer('conv3x3_18', convBlock(3,3,512,512), {'x18'}, {'x19'}, {'f18', 'b18'});
    net.addLayer('relu_19', dagnn.ReLU(), {'x19'}, {'x20'}, {});
    net.addLayer('unpool_20', dagnn.Unpooling(), {'x20'}, {'x21'}, {});
    
    net.addLayer('conv5x5_21', convBlock(5,5,512,256), {'x21'},{'x22'}, {'f21','b21'});
    net.addLayer('crop_22', dagnn.Crop(), {'x15', 'x22'}, {'x23'}, {});
    net.addLayer('concat_23', dagnn.Concat('dim', 3), {'x23', 'x22'}, {'x24'}, {});
    net.addLayer('conv3x3_24', convBlock(3,3,512,256), {'x24'}, {'x25'}, {'f24', 'b24'});
    net.addLayer('relu_25', dagnn.ReLU(), {'x25'}, {'x26'}, {});
    net.addLayer('conv3x3_26', convBlock(3,3,256,256), {'x26'}, {'x27'}, {'f26', 'b26'});
    net.addLayer('relu_27', dagnn.ReLU(), {'x27'}, {'x28'}, {});
    net.addLayer('unpool_28', dagnn.Unpooling(), {'x28'}, {'x29'}, {});
    
    net.addLayer('conv5x5_29', convBlock(5,5,256,128), {'x29'},{'x30'}, {'f29','b29'});
    net.addLayer('crop_30', dagnn.Crop(), {'x10', 'x30'}, {'x31'}, {});
    net.addLayer('concat_31', dagnn.Concat('dim', 3), {'x31', 'x30'}, {'x32'}, {});
    net.addLayer('conv3x3_32', convBlock(3,3,256,128), {'x32'}, {'x33'}, {'f32', 'b32'});
    net.addLayer('relu_33', dagnn.ReLU(), {'x33'}, {'x34'}, {});
    net.addLayer('conv3x3_34', convBlock(3,3,128,128), {'x34'}, {'x35'}, {'f34', 'b34'});
    net.addLayer('relu_35', dagnn.ReLU(), {'x35'}, {'x36'}, {});
    net.addLayer('unpool_36', dagnn.Unpooling(), {'x36'}, {'x37'}, {});
    
    net.addLayer('conv5x5_37', convBlock(5,5,128,64), {'x37'},{'x38'}, {'f37','b37'});
    net.addLayer('crop_38', dagnn.Crop(), {'x05', 'x38'}, {'x39'}, {});
    net.addLayer('concat_39', dagnn.Concat('dim', 3), {'x39', 'x38'}, {'x40'}, {});
    net.addLayer('conv3x3_40', convBlock(3,3,128,64), {'x40'}, {'x41'}, {'f40', 'b40'});
    net.addLayer('relu_41', dagnn.ReLU(), {'x41'}, {'x42'}, {});
    net.addLayer('conv3x3_42', convBlock(3,3,64,64), {'x42'}, {'x43'}, {'f42', 'b42'});
    net.addLayer('relu_43', dagnn.ReLU(), {'x43'}, {'x44'}, {});
    net.addLayer('conv1x1_44', convBlock(1,1,64,1), {'x44'}, {'x45'}, {'f44', 'b44'});
    
    net.addLayer('loss', dagnn.Loss('loss', 'regloss'), {'x45', 'labels'}, 'objective') ;

    %Initialise random parameters
    net.initParams;
    
    
    
    %Visualize Network (372 -> 256)
    %net.print({'input', [372 372 1]}, 'all', true, 'format', 'dot')
    
    %Receptive Fields
    %net.getVarReceptiveFields('x01').size
end

function convObj = convBlock(fh,fw,fc,k)
    convObj = dagnn.Conv('size', [fh fw fc k], 'hasBias', true);
end