function convtObj = convtBlock(fh,fw,k,fd)
    convtObj = dagnn.ConvTranspose('size', [fh fw k fd], 'upsample', 2, 'hasBias', true);
end