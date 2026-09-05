import math
import unittest
import trainlab as lab

class TrainingTests(unittest.TestCase):
    def test_sigmoid_extremes(self):
        self.assertEqual(lab.sigmoid(-1000),0)
        self.assertEqual(lab.sigmoid(1000),1)
        self.assertEqual(lab.sigmoid(0),0.5)


    def test_seed(self):
        self.assertEqual(lab.dataset(seed=3),lab.dataset(seed=3))
        self.assertNotEqual(lab.dataset(seed=3),lab.dataset(seed=4))


    def test_split(self):
        rows,labels=[[i] for i in range(20)],[i%2 for i in range(20)]
        a,b,c,d=lab.split(rows,labels)
        self.assertEqual(len(a)+len(c),20)
        self.assertFalse({r[0] for r in a}&{r[0] for r in c})
        for r,y in zip(a+c,b+d):
            self.assertEqual(r[0]%2,y)


