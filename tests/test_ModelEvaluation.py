import unittest 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from validation.Holdout import Holdout
from validation.LeaveOneOut import LeaveOneOut
from validation.RandomSubsampling import RandomSubsampling
from validation.ModelEvaluation import ModelEvaluationFactory
from validation.model_selection import ModelSelection

class TestModelSelection(unittest.TestCase):
    
    def test_holdout_instance_from_factory(self):
        """
        Verifica che scegliendo "1" la factory restituisca un oggetto Holdout
        e che il test_size sia impostato correttamente.
        """
        strategy = ModelEvaluationFactory.get_validation_strategy("1", test_size=0.25)
        self.assertIsInstance(strategy, Holdout)
        self.assertEqual(strategy.test_size, 0.25)

    def test_leave_one_out_instance_from_factory(self):
        """
        Verifica che scegliendo "2" la factory restituisca un oggetto LeaveOneOut
        e che K sia impostato correttamente.
        """
        strategy = ModelEvaluationFactory.get_validation_strategy("2", K=5)
        self.assertIsInstance(strategy, LeaveOneOut)
        self.assertEqual(strategy.K, 5)

    def test_random_subsampling_instance_from_factory(self):
        """
        Verifica che scegliendo "3" la factory restituisca un oggetto RandomSubsampling
        e che test_size e num_splits siano impostati correttamente.
        """
        strategy = ModelEvaluationFactory.get_validation_strategy("3", test_size=0.3, num_splits=10)
        self.assertIsInstance(strategy, RandomSubsampling)
        self.assertEqual(strategy.test_size, 0.3)
        self.assertEqual(strategy.num_splits, 10)

    # verifichiamo che si sollevi un'eccezione correttamente
    def test_invalid_choice_raises_value_error(self):
        """
        Verifica che un valore non valido per choice sollevi ValueError.
        """
        with self.assertRaises(ValueError):
            a = ModelEvaluationFactory.get_validation_strategy("invalid_choice")


if __name__ == '__main__': #il codice viene eseguito solo se il file viene lanciato direttamente come script principale
    unittest.main()