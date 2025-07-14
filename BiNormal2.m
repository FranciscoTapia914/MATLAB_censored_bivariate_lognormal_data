% File: Normal.m
classdef BiNormal2 < handle
    %NORMAL   Bivariate normal with censoring and likelihood calculation

    properties
        data                 % ndata×2 array of [logX, logY]
        ndata                % number of observations
        NVAR                 % number of variables (2)
        coef0                % constant term in log‐MVN pdf

        rho                  % correlation coefficient
        stdX                 % standard deviation of X
        stdY                 % standard deviation of Y
        covMat               % 2×2 covariance matrix
        meanVec              % 1×2 mean vector [meanX, meanY]
        stdRatio             % stdX/stdY
        stdXGivenY           % conditional std of X given Y
        cutoffIntercept      % intercept of censoring line
        cutoffSlopeInverse   % 1/slope of censoring line
        lowerLimY            % integration lower limit for Y
        upperLimY            % integration upper limit for Y
    end

    methods
        function obj = BiNormal2(logX, logY)
            % Constructor: check lengths, store data, precompute coef0.
            if numel(logX) ~= numel(logY)
                error('Error: dataset lengths not equivalent');
            end
            obj.data   = double([logX(:), logY(:)]);
            obj.ndata  = size(obj.data,1);
            obj.NVAR   = 2;
            obj.coef0  = obj.NVAR * log(1/sqrt(2*pi));
        end

        function val = integrateXGivenY(obj, y)
            % Integrand: f_Y(y) * P[X > threshold | Y=y]
            avgXGivenY = obj.meanVec(1) ...
                       + obj.stdRatio * obj.rho * (y - obj.meanVec(2));
            threshX    = (y - obj.cutoffIntercept) * obj.cutoffSlopeInverse;
            val = normpdf(y, obj.meanVec(2), obj.stdY) ...
                .* normcdf(threshX, avgXGivenY, obj.stdXGivenY);
        end

        function normFac = getNormFac(obj, vecValued)
            % Normalization factor via numerical integration
            if nargin < 2
                vecValued = false;
            end
            opts = {'AbsTol',1e-4};
            if vecValued
                opts = [opts, {'ArrayValued',true}];
            end
            normFac = integral( @(y) obj.integrateXGivenY(y), ...
                                 obj.lowerLimY, obj.upperLimY, opts{:} );
        end

        function logLike = getLogLike_fast(obj, param)
            % Compute log‐likelihood for given parameter vector
            % param = [meanX, meanY, logStdX, logStdY, fisherRho, slope, intercept]

            % Ensure param is a 1×7 row vector
            param = reshape(param, 1, []);

            % Unpack and compute basic stats
            obj.rho     = tanh( param(5) );
            obj.meanVec = double( param(1:2) );
            obj.stdX    = exp( param(3) );
            obj.stdY    = exp( param(4) );

            % Build covariance matrix
            obj.covMat = [ obj.stdX^2,                obj.rho*obj.stdX*obj.stdY;
                           obj.rho*obj.stdX*obj.stdY, obj.stdY^2             ];

            % Conditional and scaling terms
            obj.stdXGivenY         = sqrt( obj.covMat(1,1) * (1 - obj.rho^2) );
            obj.stdRatio           = obj.stdX / obj.stdY;
            obj.cutoffSlopeInverse = 1/param(6);
            obj.cutoffIntercept    = param(7);

            % Integration limits for Y (±3σ)
            dY = 3 * obj.stdY;
            obj.lowerLimY = obj.meanVec(2) - dY;
            obj.upperLimY = obj.meanVec(2) + dY;

            % Check covariance positive‐definite
            if det(obj.covMat) == 0
                error('Covariance matrix not positive‐definite.');
            end

            invCovMat = inv(obj.covMat);
            coef = obj.coef0 + log( sqrt(det(invCovMat)) );

            % Censoring‐line check
            X = obj.data(:,1);
            Y = obj.data(:,2);
            slope     = param(6);
            intercept = param(7);
            if any( Y < slope*X + intercept )
                logLike = -1e100;
                return
            end

            % Mahalanobis‐like term
            diffs = bsxfun(@minus, obj.data, obj.meanVec);   % ndata×2
            quad  = sum( (diffs * invCovMat) .* diffs, 2 );  % ndata×1

            % Normalizing constant (scalar)
            normFac = obj.getNormFac(false);

            % Final log‐likelihood
            logLike = obj.ndata * (coef - log(normFac)) ...
                      - 0.5 * sum(quad);
        end
    end
end