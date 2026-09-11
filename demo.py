"""Run from the repository root: python demo.py --seed 7."""
import argparse
import json
from trainlab import dataset,split,fit,predict,accuracy,loss

def run(seed=7,epochs=300):
    rows,labels=dataset(seed=seed)
    a,b,c,d=split(rows,labels,seed=seed+1)
    w,bias=fit(a,b,epochs=epochs)
    probabilities=predict(c,w,bias)
    return {"seed":seed,"epochs":epochs,"train_samples":len(a),"test_samples":len(c),
            "test_accuracy":accuracy(d,probabilities),"test_loss":loss(d,probabilities)}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--epochs",type=int,default=300)
    args=parser.parse_args()
    print(json.dumps(run(args.seed,args.epochs),indent=2))
