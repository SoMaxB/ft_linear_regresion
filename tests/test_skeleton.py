import unittest

import predict
import train


class SkeletonTest(unittest.TestCase):
    def test_entry_points_expose_main(self):
        self.assertTrue(callable(train.main))
        self.assertTrue(callable(predict.main))


if __name__ == "__main__":
    unittest.main()
