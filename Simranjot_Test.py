import Simranjot_Circulation as Circulation
from scipy.signal import find_peaks

# Simulate model in Simranjot_Circulation
model = Circulation.Circulation(75, 2.0, 0.06) # Change the first value for Circulation.Circulation to either 75 or 175 for pytest
time,state = model.simulate(10)

def check_changing_pressures(states):
    """
        :param states: [x1,x2,x3,x4] retrieved from model.simulate
    """
    peaks = find_peaks(states[2], height = (1,200))

    # Only use peaks from middle of simulation onwards since first few peaks of simulation tend to be unstable
    middle_of_list = int(len(peaks[0])/2)
    peak_heights = peaks[1]['peak_heights'][middle_of_list:]
    dif_in_peak_heights = []

    # Create list of difference in height between consecutive peaks from midpoint onwards
    for i in range(len(peak_heights)-1):
        dif_in_peak_heights.append(abs(peak_heights[i] - peak_heights[i+1]))
    
    if any (x > 0.1 for x in dif_in_peak_heights):
        return False
    else:
        return True

def test_changing_pressures():
    # 75 bpm should passes while 175 bpm fails
    assert check_changing_pressures(state) == True, 'Changing peak pressure detected -- Simulation fail'