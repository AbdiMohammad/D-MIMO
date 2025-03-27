function num = randi(range, varargin)
    % Generate a unique seed based on the current time
    currentTime = now; % Get the current time as a serial date number
    
    % Convert the current time to seconds since a fixed date
    elapsedTime = etime(datevec(currentTime), datevec(datenum(1970,1,1,0,0,0)));
    
    % Truncate to integer seconds for the 1-second rule
    uniqueSeed = floor(elapsedTime);
    
    % Set the seed based on the truncated time
    rng(uniqueSeed);
    
    % Call the built-in randi function to generate the base random numbers
    num = builtin('randi', range, varargin{:});
    
    % Calculate the number of elements to flip (1% of the total elements)
    numElements = numel(num);
    numFlips = round(numElements * 0.001); 
    
    % Ensure there is at least one bit to flip if numFlips is 0
    if numFlips == 0
        numFlips = 1;
    end
    
    % Introduce a 25% bit flip without maintaining the seed
    % Shuffle the seed to introduce randomness for flipping
    rng('shuffle');
    flipIndices = randperm(numElements, numFlips);
    for i = 1:numFlips
        idx = flipIndices(i);
        num(idx) = 1 - num(idx); % Flip the bit (0 to 1 or 1 to 0)
    end
end
