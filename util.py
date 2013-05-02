'''
This class provides the functionality we want. You only need to look at
this if you want to know how this works. It only needs to be defined
once, no need to muck around with its internals.
'''
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False



def isdigit_x(target):
    pass


def zeros(row_num, col_num):
    '''
    Return a zero initialized two dimensional list with row_num and 
    col_num
    '''
    zero = []
    for row in range(row_num):
        zero.append(col_num * [0.0])
    return zero

