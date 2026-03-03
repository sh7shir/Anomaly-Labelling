"""
SLIDING WINDOW AND AGEING FUNCTIONS

These functions calculate the beginning of the sliding window and ageing coefficients for AnDePeD.
"""


# calculating window
def update_window_beginning(self, time):
    if time - self.WINDOW_SIZE + 1 <= self.B:
        self.window_beginning = self.B
    else:
        self.window_beginning = time - self.WINDOW_SIZE + 1


# calculating the ageing coefficient
def ageing_coefficient(self, time, y):
    if self.USE_AGING:
        if time == self.window_beginning:
            return 1
        else:
            return ((y - self.window_beginning) / (time - self.window_beginning)) ** self.AGE_POWER
    else:
        return 1
