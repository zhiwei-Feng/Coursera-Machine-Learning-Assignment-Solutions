import matplotlib.pyplot as plt


def plot_data(x, y):
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Plot the training data into a figure using the
    %               "figure" and "plot" commands. Set the axes labels using
    %               the "xlabel" and "ylabel" commands. Assume the
    %               population and revenue data have been passed in
    %               as the x and y arguments of this function.
    %
    % Hint: You can use the 'rx' option with plot to have the markers
    %       appear as red crosses. Furthermore, you can make the
    %       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
    """
    plt.plot(x, y, 'rx',label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')