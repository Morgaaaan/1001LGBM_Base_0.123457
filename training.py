# training.py  (LightGBM 4.x compatible; outputs to ./output by default)
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import lightgbm as lgb
from model import ModelLGBM

def load_feature_spec(path:str)->dict:
    with open(path,"r",encoding="utf-8") as f: obj=json.load(f)
    feats=obj.get("features",{})
    return {
        "numeric":feats.get("numeric",[]),
        "categorical":feats.get("categorical",[]),
        "label":obj.get("label"),
        "id_cols":obj.get("id_cols",[]),
        "drop":obj.get("drop",[]),
    }

def apply_spec(df:pd.DataFrame,spec:dict,for_training=True):
    df=df.copy()
    for col in spec.get("drop",[]) or []:
        if col in df.columns: df.drop(columns=[col],inplace=True)
    for col in spec.get("categorical",[]) or []:
        if col in df.columns: df[col]=df[col].astype("category")
    feature_cols=(spec.get("numeric",[]) or [])+(spec.get("categorical",[]) or [])
    X=df[feature_cols]; y=None
    if for_training and (spec.get("label") in df.columns):
        y=df[spec["label"]]
    return X,y

def parse_args():
    ap=argparse.ArgumentParser(description="Train LightGBM (sklearn API, LGBM 4.x)")
    ap.add_argument("--train_csv",required=True)
    ap.add_argument("--valid_csv")
    ap.add_argument("--feature_spec",default="./feature_spec.json")
    ap.add_argument("--output_dir",default="./output/lgbm_v1")   # ← 改成 output
    # LGBM params
    ap.add_argument("--learning_rate",type=float,default=0.05)
    ap.add_argument("--n_estimators",type=int,default=5000)
    ap.add_argument("--num_leaves",type=int,default=63)
    ap.add_argument("--max_depth",type=int,default=-1)
    ap.add_argument("--min_child_samples",type=int,default=100)
    ap.add_argument("--colsample_bytree",type=float,default=0.7)
    ap.add_argument("--subsample",type=float,default=0.8)
    ap.add_argument("--subsample_freq",type=int,default=1)
    ap.add_argument("--reg_alpha",type=float,default=0.0)
    ap.add_argument("--reg_lambda",type=float,default=0.0)
    ap.add_argument("--max_bin",type=int,default=127)
    ap.add_argument("--scale_pos_weight",type=float,default=None)
    ap.add_argument("--device",default=None)
    # training control
    ap.add_argument("--early_stopping_rounds",type=int,default=200)
    ap.add_argument("--eval_metric",default="auc")
    ap.add_argument("--verbose",type=int,default=50)
    return ap.parse_args()

def main():
    args=parse_args()
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    spec=load_feature_spec(args.feature_spec)

    # load data
    df_train=pd.read_csv(args.train_csv); X_train,y_train=apply_spec(df_train,spec,True)
    X_valid=y_valid=None
    eval_set=None
    if args.valid_csv:
        df_valid=pd.read_csv(args.valid_csv); X_valid,y_valid=apply_spec(df_valid,spec,True)
        eval_set=[(X_valid,y_valid)]

    # build model
    model=ModelLGBM(
        learning_rate=args.learning_rate,n_estimators=args.n_estimators,num_leaves=args.num_leaves,
        max_depth=args.max_depth,min_child_samples=args.min_child_samples,colsample_bytree=args.colsample_bytree,
        subsample=args.subsample,subsample_freq=args.subsample_freq,reg_alpha=args.reg_alpha,reg_lambda=args.reg_lambda,
        max_bin=args.max_bin,scale_pos_weight=args.scale_pos_weight,device=args.device
    )

    # ---- LightGBM 4.x: use callbacks instead of early_stopping_rounds/evals_result ----
    callbacks=[]
    eval_history={}
    if eval_set is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False))
        callbacks.append(lgb.record_evaluation(eval_history))
        callbacks.append(lgb.log_evaluation(period=args.verbose))

    # set metric to underlying sklearn model
    model.model.set_params(eval_metric=args.eval_metric)

    # fit
    model.model.fit(
        X_train,y_train,
        eval_set=eval_set,
        callbacks=callbacks,
    )

    # metrics & curves
    if X_valid is not None:
        probs=model.predict_proba(X_valid); preds=(probs>=0.5).astype(int)
        auc=roc_auc_score(y_valid,probs); ap=average_precision_score(y_valid,probs); f1=f1_score(y_valid,preds)
        print(f"[VALID] ROC-AUC={auc:.6f} | PR-AUC(AP)={ap:.6f} | F1@0.5={f1:.6f}")

        # save eval curves
        if "valid_0" in eval_history and len(eval_history["valid_0"])>0:
            curve=pd.DataFrame(eval_history["valid_0"])
            curve.to_csv(out_dir/"training_curve.csv",index_label="iteration")
            plt.figure()
            for m,vals in eval_history["valid_0"].items():
                plt.plot(vals,label=m)
            plt.xlabel("Iteration"); plt.ylabel("Metric value"); plt.legend()
            plt.title("Validation metrics"); plt.savefig(out_dir/"training_curve.png")
            print(f"[INFO] saved: {out_dir/'training_curve.csv'}, {out_dir/'training_curve.png'}")

    # save model
    model.save(out_dir/"model.pkl")
    print(f"[INFO] model saved to {out_dir/'model.pkl'}")

if __name__=="__main__":
    main()
