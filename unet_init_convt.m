function net = unet_init_convt()
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
    net.addLayer('convT2x2_20', convtBlock(2,2,256,512), {'x20'}, {'x21'}, {'uf20', 'ub20'});
    
    net.addLayer('crop_21', dagnn.Crop(), {'x15', 'x21'}, {'x22'}, {});
    net.addLayer('concat_22', dagnn.Concat('dim', 3), {'x22', 'x21'}, {'x23'}, {});
    net.addLayer('conv3x3_23', convBlock(3,3,512,256), {'x23'}, {'x24'}, {'f23', 'b23'});
    net.addLayer('relu_24', dagnn.ReLU(), {'x24'}, {'x25'}, {});
    net.addLayer('conv3x3_25', convBlock(3,3,256,256), {'x25'}, {'x26'}, {'f25', 'b25'});
    net.addLayer('relu_26', dagnn.ReLU(), {'x26'}, {'x27'}, {});
    net.addLayer('convT2x2_27', convtBlock(2,2,128,256), {'x27'}, {'x28'}, {'uf27', 'ub27'});
    
    net.addLayer('crop_28', dagnn.Crop(), {'x10', 'x28'}, {'x29'}, {});
    net.addLayer('concat_29', dagnn.Concat('dim', 3), {'x29', 'x28'}, {'x30'}, {});
    net.addLayer('conv3x3_30', convBlock(3,3,256,128), {'x30'}, {'x31'}, {'f30', 'b30'});
    net.addLayer('relu_31', dagnn.ReLU(), {'x31'}, {'x32'}, {});
    net.addLayer('conv3x3_32', convBlock(3,3,128,128), {'x32'}, {'x33'}, {'f32', 'b32'});
    net.addLayer('relu_33', dagnn.ReLU(), {'x33'}, {'x34'}, {});
    net.addLayer('convT2x2_34', convtBlock(2,2,64,128), {'x34'}, {'x35'}, {'uf34', 'ub34'});
    
    net.addLayer('crop_35', dagnn.Crop(), {'x05', 'x35'}, {'x36'}, {});
    net.addLayer('concat_36', dagnn.Concat('dim', 3), {'x36', 'x35'}, {'x37'}, {});
    net.addLayer('conv3x3_37', convBlock(3,3,128,64), {'x37'}, {'x38'}, {'f37', 'b37'});
    net.addLayer('relu_38', dagnn.ReLU(), {'x38'}, {'x39'}, {});
    net.addLayer('conv3x3_39', convBlock(3,3,64,64), {'x39'}, {'x40'}, {'f39', 'b39'});
    net.addLayer('relu_40', dagnn.ReLU(), {'x40'}, {'x41'}, {});
    net.addLayer('conv1x1_41', convBlock(1,1,64,1), {'x41'}, {'x42'}, {'f41', 'b41'});
    
    net.addLayer('loss', dagnn.Loss('loss', 'regloss'), {'x42', 'labels'}, 'objective') ;

    %Initialise random parameters
    net.initParams();
    
    
    
    %Visualize Network (340 -> 252)
    net.print({'input', [340 340 1]}, 'all', true, 'format', 'dot')
    
    %Receptive Fields
    %net.getVarReceptiveFields('x01').size
end

function convObj = convBlock(fh,fw,fc,k)
    convObj = dagnn.Conv('size', [fh fw fc k], 'hasBias', true);
end

function convtObj = convtBlock(fh, fw, k, fc)
    convtObj = dagnn.ConvTranspose('size', [fh fw k fc], 'upsample', 2, 'hasBias', true);
end