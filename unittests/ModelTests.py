#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from model import *

model_name = re.sub("\.","_",str(MODEL_VERSION))


class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(test=True, clean=False)
        self.assertTrue(os.path.exists(os.path.join("models", "test-all-{}.joblib".format(model_name))))

    def test_02_load(self):
        """
        test the load functionality
        """
                        
        ## train the model
        all_data, all_models = model_load(prefix='test')
        
        self.assertTrue('predict' in dir(all_models['all']))
        self.assertTrue('fit' in dir(all_models['all']))
        self.assertTrue('predict' in dir(all_models['united_kingdom']))
        self.assertTrue('fit' in dir(all_models['united_kingdom']))
       
    def test_03_predict(self):
        """
        test the predict function input
        """
    
        ## ensure that a list can be passed
        query = {'country': 'united_kingdom',
                'year': '2019',
                'month': '05',
                'day': '15'
        }

        result = model_predict(query['country'],
                               query['year'] ,
                               query['month'],
                               query['day'], test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred > 0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
