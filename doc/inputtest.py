
from inputmodel import *
import sys
import filecmp

if __name__=='__main__':
    InputFile = "input_example.json"
    OutputFile = "input_example_re.json"
    model = Model()
    model.loadFromFile(InputFile)    
    model.saveToFile(OutputFile)
    
    #if filecmp.cmp(InputFile, OutputFile, shallow=False):
    #  print "Test OK!"
    #else:
    #  print "TEST FAILED. Files are different."