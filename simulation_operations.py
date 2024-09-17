
import numpy as np

class simulation_oper:
    def __init__(self, all_simulations):
        """ numpy array matrix, each column are the simulated returns for the same instrument"""
        self.all_simulations = all_simulations

    def average_returns(self):
        """the array of average monthly returns for all simulations"""
        return np.mean(self.all_simulations, axis = 0)
    
    def check_validity(self, low, high):
        """low is the theoretical lower bound """
        """high is the theoretical higher bound """
        
        average_simulation_returns = self.average_returns()
        
        mask = (average_simulation_returns >= low) & (average_simulation_returns <= high)


        
        percent = np.sum(mask) / len(average_simulation_returns) * 100
        
        return percent 
    
    def get_log_returns(self):
        """ convert simple returns to log returns in one step"""
        self.log_returns = np.log( 1 + self.all_simulations)
        return self.log_returns