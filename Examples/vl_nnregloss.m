function Y = vl_nnregloss(X, c, dzdy, varargin)

% VL_NNREGLOSS  CNN regression loss
%    Y = VL_NNREGLOSS(X, l) applies a regression loss to the data X. 
%    X has dimension H x W x D x N, packing N arrays of W x H
%    D-dimensional vectors.
%
%    l contains the ground truth values. l has dimensions H x W x 1 x N,
%    such that every location (pixel) is assigned an expected value.
%
%    Often W=H=1, but this is not a requirement, as the operator is applied
%    convolutionally at all spatial locations. In this case, D can be 
%    thought of as the number of locations (pixels) at which an output 
%    value is to be calculated. 
%
%    DZDX = VL_NNLOSS(, C, DZDY) computes the derivative DZDX of the
%    CNN with respect to the input X given the derivative DZDY with
%    respect to the block output Y. DZDX has the same dimension as X.
%
%    The loss function itself is an average (adjusted by lambda) between
%    L2-loss and the scale-invariant error of Eigen et al.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

diff = X - c;
if (nargin <= 2)||isempty(dzdy)  
    Y =  sum( reshape(diff, [], size(c, 4)) .^ 2, 1); %forward-propagation
else
    Y = 2 .* diff .* dzdy;     %back to initial form
    %Y = sign(diff);
end


end