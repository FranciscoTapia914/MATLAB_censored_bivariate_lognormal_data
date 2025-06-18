classdef BiNormal2
    properties
        data
        ndata
        NVAR = 2
        coef0
        rho
        stdX
        stdY
        covMat
        meanVec
        stdRatio
        stdXGivenY
        cutoffIntercept
        cutoffSlopeInverse
        lowerLimY
        upperLimY
        invCovMat
    end
    
    methods
        function obj = BiNormal2(logX, logY)
            if length(logX) ~= length(logY)
                error("Error: dataset lengths not equivalent");
            end
            
            obj.data = double([logX(:), logY(:)]);
            obj.ndata = length(logX);
            obj.coef0 = obj.NVAR * log(1 / sqrt(2 * pi));
        end
        
        function val = integrateXGivenY(obj, yval)
            avgXGivenY = obj.meanVec(1) + obj.stdRatio * obj.rho * (yval - obj.meanVec(2));
            threshX = (yval - obj.cutoffIntercept) * obj.cutoffSlopeInverse;
            val = normpdf(yval, obj.meanVec(2), obj.stdY) * normcdf(threshX, avgXGivenY, obj.stdXGivenY);
        end
        
        function normFac = getNormFac(obj)
            normFac = integral(@(y) obj.integrateXGivenY(y), obj.lowerLimY, obj.upperLimY, 'AbsTol', 1e-4);
        end
        
        function logLike = getLogLike_fast(obj, param)
            obj.rho = tanh(param(5));
            obj.meanVec = param(1:2);
            obj.stdX = exp(param(3));
            obj.stdY = exp(param(4));
            obj.cutoffSlopeInverse = 1 / param(6);
            obj.cutoffIntercept = param(7);

            obj.covMat = [obj.stdX^2, obj.rho * obj.stdX * obj.stdY;
                  obj.rho * obj.stdX * obj.stdY, obj.stdY^2];

            obj.stdXGivenY = sqrt(obj.stdX^2 * (1 - obj.rho^2));
            obj.stdRatio = obj.stdX / obj.stdY;
            %obj.cutoffSlopeInverse = 1 / param(6);
            %obj.cutoffIntercept = param(7);

            significance = 4 * obj.stdY;
            obj.lowerLimY = obj.meanVec(2) - significance;
            obj.upperLimY = obj.meanVec(2) + significance;

            disp(size(obj.data));
            disp(size(obj.meanVec));
            disp(size(obj.covMat));

            if det(obj.covMat) == 0
                error("Covariance matrix not positive-definite.");
            end

            obj.invCovMat = inv(obj.covMat);

            disp(size(obj.invCovMat));

            coef = -0.5*obj.NVAR * log(2*pi) - 0.5*log(det(obj.invCovMat));

            disp(size(coef));

            % Vectorized condition check
            invalidIdx = obj.data(:,2) < (param(6) * obj.data(:,1) + param(7));
            if any(invalidIdx)
                logLike = -1e100;
                return;
            end

            % Vectorized quadratic term computation
            normedPoints = obj.data - obj.meanVec;

            disp(size(normedPoints));

            quadraticTerms = sum(normedPoints * obj.invCovMat * normedPoints');

            disp(size(quadraticTerms));

            logLike = sum(coef - 0.5 * quadraticTerms);

            disp(size(logLike));

            % Approximate normalization factor
            yvals = linspace(obj.lowerLimY, obj.upperLimY, 100);
            normFac = sum(arrayfun(@obj.integrateXGivenY, yvals)) * (obj.upperLimY - obj.lowerLimY) / 100;

            logLike = logLike - obj.ndata * log(normFac);
        end
    end
end