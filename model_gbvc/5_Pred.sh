#!/bin/bash

pred_src=data_pred/pred_src
pred_res=data_pred/pred_res
cat labeling/unlabeled/data_cln.tsv | sort -R | head -1000 > $pred_src
python 5_Pred.py $pred_src | sort -t$'\t' -k1,1gr > $pred_res
cat $pred_res | awk -F'\t' '{if($1>0.4 && $1<0.6) print $2}' > labeling/unlabeled/iter2_boundary

