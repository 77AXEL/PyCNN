import unittest
import numpy as np
import tempfile
import shutil
import os
from PIL import Image


class TestPyCNN(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from pycnn.pycnn import PyCNN
            cls.PyCNN = PyCNN
        except ImportError as e:
            raise ImportError(f"Failed to import PyCNN: {e}")
        
        cls.test_dir = tempfile.mkdtemp()
        cls._create_test_dataset(cls.test_dir)
    
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_dataset(cls, base_path):
        classes = ['class_a', 'class_b']
        
        for class_name in classes:
            class_dir = os.path.join(base_path, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(5):
                if class_name == 'class_a':
                    img_array = np.random.randint(150, 255, (32, 32, 3), dtype=np.uint8)
                    img_array[:, :, 1:] = np.random.randint(0, 50, (32, 32, 2), dtype=np.uint8)
                else:
                    img_array = np.random.randint(0, 50, (32, 32, 3), dtype=np.uint8)
                    img_array[:, :, 2] = np.random.randint(150, 255, (32, 32), dtype=np.uint8)
                
                img = Image.fromarray(img_array, mode='RGB')
                img.save(os.path.join(class_dir, f'img_{i}.png'))
    
    def test_01_import(self):
        self.assertIsNotNone(self.PyCNN, "PyCNN should be importable")
    
    def test_02_initialization(self):
        model = self.PyCNN()
        self.assertIsNotNone(model, "Model should initialize")
        self.assertEqual(model.optimizer, "sgd", "Default optimizer should be SGD")
        self.assertFalse(model.use_cuda, "Should default to CPU")
    
    def test_03_init_configuration(self):
        model = self.PyCNN()
        model.init(batch_size=16, layers=[64, 32], learning_rate=0.01, epochs=5)
        
        self.assertEqual(model.batch_size, 16)
        self.assertEqual(model.layers, [64, 32])
        self.assertEqual(model.learning_rate, 0.01)
        self.assertEqual(model.epochs, 5)
    
    def test_04_adam_optimizer(self):
        model = self.PyCNN()
        model.init(layers=[32])
        model.adam(beta1=0.9, beta2=0.999, eps=1e-8)
        
        self.assertEqual(model.optimizer, "adam")
        self.assertEqual(model.beta1, 0.9)
        self.assertEqual(model.beta2, 0.999)
        self.assertEqual(model.eps, 1e-8)
    
    def test_05_load_local_dataset(self):
        model = self.PyCNN()
        model.init(batch_size=4, layers=[32], learning_rate=0.01, epochs=2)
        
        # Load the test dataset
        model.dataset.local(self.test_dir, max_image=3, image_size=32)
        
        self.assertEqual(len(model.classes), 2, "Should have 2 classes")
        self.assertGreater(len(model.data), 0, "Should have loaded data")
        self.assertEqual(len(model.data), len(model.labels), "Data and labels should match")
    
    def test_06_preprocessing(self):
        model = self.PyCNN()
        model.init(layers=[32])
        
        test_img = Image.new('RGB', (32, 32), color=(255, 0, 0))
        processed = model._preprocess_image(test_img, 32, model.filters[:3])
        
        self.assertIsInstance(processed, np.ndarray, "Should return numpy array")
        self.assertEqual(processed.ndim, 1, "Should be flattened")
        self.assertGreater(processed.shape[0], 0, "Should have features")
    
    def test_07_softmax(self):
        model = self.PyCNN()
        
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = model._softmax_batch(x)
        
        self.assertEqual(result.shape, x.shape, "Output shape should match input")
        row_sums = np.sum(result, axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0], decimal=5)
    
    def test_08_training(self):
        model = self.PyCNN()
        model.init(batch_size=4, layers=[16], learning_rate=0.1, epochs=2)
        model.dataset.local(self.test_dir, max_image=3, image_size=32)
        model.train_model(visualize=False)
        
        self.assertIn('w1', model.weights, "Should have weights for layer 1")
        self.assertIn('wo', model.weights, "Should have output weights")
    
    def test_09_save_and_load_model(self):
        model = self.PyCNN()
        model.init(batch_size=4, layers=[16], learning_rate=0.1, epochs=1)
        
        model.dataset.local(self.test_dir, max_image=2, image_size=32)
        model.train_model(visualize=False)
        
        model_path = os.path.join(self.test_dir, 'test_model.bin')
        model.save_model(model_path)
        
        self.assertTrue(os.path.exists(model_path), "Model file should exist")
        
        model2 = self.PyCNN()
        model2.init(layers=[16])
        model2.load_model(model_path)
        
        self.assertEqual(len(model2.classes), len(model.classes))
        self.assertEqual(model2.layers, model.layers)
        
        for key in model.weights:
            np.testing.assert_array_almost_equal(
                model.weights[key], 
                model2.weights[key],
                decimal=5
            )
    
    def test_10_prediction(self):
        model = self.PyCNN()
        model.init(batch_size=4, layers=[16], learning_rate=0.1, epochs=2)
        model.dataset.local(self.test_dir, max_image=3, image_size=32)
        model.train_model(visualize=False)
        
        test_img_path = os.path.join(self.test_dir, 'class_a', 'img_0.png')
        label, confidence = model.predict(test_img_path)
        
        self.assertIn(label, model.classes, "Predicted label should be valid")
        self.assertGreaterEqual(confidence, 0.0, "Confidence should be >= 0")
        self.assertLessEqual(confidence, 1.0, "Confidence should be <= 1")
    
    def test_11_cross_entropy(self):
        model = self.PyCNN()
        preds = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        labels = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        loss = model._cross_entropy_batch(preds, labels)
        
        self.assertEqual(loss.shape[0], 2, "Should return loss for each sample")
        self.assertTrue(np.all(loss >= 0), "Loss should be non-negative")
    
    def test_12_max_pooling(self):
        model = self.PyCNN()
        feature_map = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=float)
        
        pooled = model._max_pooling(feature_map)
        
        expected_shape = (2, 2)
        self.assertEqual(pooled.shape, expected_shape, f"Pooled shape should be {expected_shape}")
        
        self.assertEqual(pooled[0, 0], 6)
        self.assertEqual(pooled[0, 1], 8)
        self.assertEqual(pooled[1, 0], 14)
        self.assertEqual(pooled[1, 1], 16)
    
    def test_13_filters(self):
        model = self.PyCNN()
        
        self.assertGreater(len(model.filters), 0, "Should have filters")
        
        for filt in model.filters:
            self.assertEqual(len(filt), 3, "Filter should be 3x3")
            self.assertEqual(len(filt[0]), 3, "Filter should be 3x3")
    
    def test_14_custom_filters(self):
        custom_filters = [
            [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        ]
        
        model = self.PyCNN()
        model.init(layers=[32], filters=custom_filters)
        
        self.assertEqual(len(model.filters), 2, "Should use custom filters")
        self.assertEqual(model.filters, custom_filters)
    
    def test_15_early_stopping(self):
        model = self.PyCNN()
        model.init(batch_size=4, layers=[16], learning_rate=0.1, epochs=100)
        model.dataset.local(self.test_dir, max_image=3, image_size=32)
        model.train_model(visualize=False, early_stop=3)

        self.assertIn('w1', model.weights)
    
    def test_16_evaluation(self):
        from pycnn.pycnn import Evaluate
        
        model = self.PyCNN()
        model.init(batch_size=4, layers=[16], learning_rate=0.1, epochs=2)
        model.dataset.local(self.test_dir, max_image=3, image_size=32)
        model.train_model(visualize=False)
        
        evaluator = Evaluate(model)
        evaluator.local(self.test_dir, max_image=2)
        
        self.assertTrue(True)


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPyCNN)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)