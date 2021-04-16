import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Circulation:
    """
    Model of systemic circulation from Ferreira et al. (2005), A Nonlinear State-Space Model
    of a Combined Cardiovascular System and a Rotary Pump, IEEE Conference on Decision and Control. 
    """

    def __init__(self, HR, Emax, Emin):
        self.set_heart_rate(HR)

        self.Emin = Emin
        self.Emax = Emax
        self.non_slack_blood_volume = 150 # ml

        self.R1 = 1.0 # between .5 and 2
        self.R2 = .005
        self.R3 = .001
        self.R4 = .0398

        self.C2 = 4.4
        self.C3 = 1.33

        self.L = .0005
        self.dt = 0.0001

    def set_heart_rate(self, HR):
        """
        Sets several related variables together to ensure that they are consistent.

        :param HR: heart rate (beats per minute)
        """
        self.HR = HR
        self.tc = 60/HR
        self.Tmax = .2+.15*self.tc # contraction time

    def get_derivative(self, t, x):
        """
        :param t: time
        :param x: state variables [ventricular pressure; atrial pressure; arterial pressure; aortic flow]
        :return: time derivatives of state variables
        """
        # Seperate state-variables from inputted state
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

        # Use the ejection-phase matrix according to Ferreira et al. for these conditions
        if (x4 > 0) or (x1 > x3):
            return np.matmul(self.ejection_phase_dynamic_matrix(t), x)
        
        # Filling phase matrix multiplication
        elif x2 > x1:
            return np.matmul(self.filling_phase_dynamic_matrix(t), x)
        
        # Isovolumic phase matrix multiplication
        else:
            return np.matmul(self.isovolumic_phase_dynamic_matrix(t), x)

        """
        WRITE CODE HERE
        Implement this by deciding whether the model is in a filling, ejecting, or isovolumic phase and using 
        the corresponding dynamic matrix. 
         
        As discussed in class, be careful about starting and ending the ejection phase. One approach is to check 
        whether the flow is >0, and another is to check whether x1>x3, but neither will work. The first won't start 
        propertly because flow isn't actually updated outside the ejection phase. The second won't end properly 
        because blood inertance will keep the blood moving briefly up the pressure gradient at the end of systole. 
        If the ejection phase ends in this time, the flow will remain non-zero until the next ejection phase. 
        """

    def isovolumic_phase_dynamic_matrix(self, t):
        """
        :param t: time (s; needed because elastance is a function of time)
        :return: A matrix for isovolumic phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt/el, 0, 0, 0],
             [0, -1/(self.R1*self.C2), 1/(self.R1*self.C2), 0],
             [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 0],
             [0, 0, 0, 0]]

    def ejection_phase_dynamic_matrix(self, t):
        """
        :param t: time (s)
        :return: A matrix for filling phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)
        return [[del_dt/el, 0, 0, -el],
                [0, -1/(self.R1*self.C2), 1/(self.R1*self.C2), 0],
                [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 1/self.C3],
                [1/self.L, 0, -1/self.L, -(self.R3+self.R4)/self.L]]

    def filling_phase_dynamic_matrix(self, t):
        """
        :param t: time (s)
        :return: A matrix for filling phase
        """
        el = self.elastance(t)
        del_dt = self.elastance_finite_difference(t)

        """
        WRITE CODE HERE
        """
        return [[(del_dt/el) - (el/self.R2), el/self.R2, 0, 0],
                [1/(self.R2 * self.C2), -(self.R1 + self.R2)/(self.C2 * self.R1 * self.R2), 1/(self.R1*self.C2), 0],
                [0, 1/(self.R1*self.C3), -1/(self.R1*self.C3), 0],
                [0, 0, 0, 0]]

    def elastance(self, t):
        """
        :param t: time (needed because elastance is a function of time)
        :return: time-varying elastance
        """
        tn = self._get_normalized_time(t)
        En = 1.55 * np.power(tn/.7, 1.9) / (1 + np.power(tn/.7, 1.9)) / (1 + np.power(tn/1.17, 21.9))
        return (self.Emax-self.Emin)*En + self.Emin

    def elastance_finite_difference(self, t):
        """
        Calculates finite-difference approximation of elastance derivative. In class I showed another method
        that calculated the derivative analytically, but I've removed it to keep things simple.

        :param t: time (needed because elastance is a function of time)
        :return: finite-difference approximation of time derivative of time-varying elastance
        """
        forward_time = t + self.dt
        backward_time = max(0, t - self.dt) # small negative times are wrapped to end of cycle
        forward = self.elastance(forward_time)
        backward = self.elastance(backward_time)
        return (forward - backward) / (2*self.dt)

    def simulate(self, total_time):
        """
        :param total_time: seconds to simulate
        :return: time, state (times at which the state is estimated, state vector at each time)
        """
        """
        WRITE CODE HERE
        Put all the blood pressure in the atria as an initial condition.
        """
        # Set the initial state with all blood pressure in atria 
        atrial_bp = self.non_slack_blood_volume/self.C2
        initial_state = [0, atrial_bp, 0, 0]

        # Solve the IVP
        solution_of_ivp = solve_ivp(self.get_derivative, (0,total_time), initial_state, max_step = self.dt) 
        
        return solution_of_ivp.t, solution_of_ivp.y

    def _get_normalized_time(self, t):
        """
        :param t: time
        :return: time normalized to self.Tmax (duration of ventricular contraction)
        """
        return (t % self.tc) / self.Tmax

    def get_aortic_pressure(self,state):
        """
        :param state: [x1,x2,x3,x4]
        :return: aortic pressure using x3,x4 and R4
        """
        return state[2] + (state[3] * self.R4)

def plot_pressures(states, time, aortic_pressure):
    """
        :param states: states returned from running the simulate function 
        :param time: time returned from running the simulate function
        :param aortic_pressure: aortic pressure returned from get_aortic_pressure function
        :return: plots of all pressures
    """
    # Plot all of the different pressures in the model
    plt.plot(time, states[0], label = 'Ventricular')
    plt.plot(time, states[1], label = 'Atrial')
    plt.plot(time, states[2], label = 'Arterial')
    plt.plot(time, aortic_pressure, label = 'Aortic')

    # Label generated plot
    plt.title('Circulatory Model')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend(loc = 'upper left')
    plt.show()

if __name__ == '__main__':
    # Part 1 - Initialize object with given parameters
    model = Circulation(75, 2.0, 0.06)

    # Part 2 - Create plots of all pressures and simulate for 5 seconds
    time, state = model.simulate(5)
    aortic_press = model.get_aortic_pressure(state)
    plot_pressures(state, time, aortic_press)
