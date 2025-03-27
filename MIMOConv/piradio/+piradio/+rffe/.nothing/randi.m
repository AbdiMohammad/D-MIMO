function num = randi(range, varargin)
    % Set the seed for reproducibility
    persistent rngState;
    if isempty(rngState)
        rng(42); % Set the seed only once
        rngState = rng; % Save the state
    else
        rng(rngState); % Restore the state
    end

    % Call the built-in randi function
    num = builtin('randi', range, varargin{:});

    % Update the state after generating random numbers
    rngState = rng;
end
