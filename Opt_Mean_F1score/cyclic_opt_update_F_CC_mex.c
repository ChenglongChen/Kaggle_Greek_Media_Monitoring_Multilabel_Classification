/*
 * Author: Chenglong Chen < yr@Kaggle >
 * Email: c.chenglong@gmail.com
 *
 */

#include "matrix.h"
#include <stdlib.h>
#include "mex.h"


// This function computes F1 score for given tp & fp & fn
double F1_score(double tp, double fp, double fn){
    double f = 0;
    double numerator = 0, denominator = 0;
    if ( (tp != 0) | (fp != 0) | (fn != 0) ){
        numerator = 2*tp;
        denominator = 2*tp + fp + fn;
        if ( denominator !=0 ){
            f = numerator / denominator;            
        }
    }
    return(f);
}


// This C-MEX programm peforms cyclic optimization
void cyclic_opt_update_F_CC_mex(double *y_valid, double *y_pred, double *wTx, double *scut_w, double *scut_b,
                                double *thresh_index, mwSize j, double *F, mwSize numValid, mwSize numLabel){
    
    // variable declaration
    double *tp, *fn, *fp;
    double *TP_j, *FN_j, *FP_j;
    double *f;
    double f_mean = 0;
    
    double *delta_y;
    double delta_wTx = 0;
	double f_ii_new = 0;
    
    // iterator
    mwSize i = 0, ii = 0;
    mwSize jj = 0, jjj = 0;
    
    /******************
     * initialization
     ******************/
    // alloc memory
    delta_y = (double *)mxCalloc(numLabel, sizeof(double));
    tp = (double *)mxCalloc(numValid, sizeof(double));
    fn = (double *)mxCalloc(numValid, sizeof(double));
    fp = (double *)mxCalloc(numValid, sizeof(double));
    TP_j = (double *)mxCalloc(numValid, sizeof(double));
    FN_j = (double *)mxCalloc(numValid, sizeof(double));
    FP_j = (double *)mxCalloc(numValid, sizeof(double));
    f = (double *)mxCalloc(numValid, sizeof(double));

    
    /*************************************
     * compute initialized mean F1-score
     *************************************/
    for (i = 0; i < numValid; i++){
        for (jj = 0; jj < numLabel; jj++){
            
            if ( (jj <= j-2) | (jj > j-1) ){
                // prior j or post j
                if ( *(y_valid + jj*numValid + i) == +1 ){
                    if ( *(y_pred + jj*numValid + i) == +1 ){                        
                        tp[i] += 1;
                        // post j
                        if (jj > j-1){
                            TP_j[i] += 1;
                        }
                    }else{
                        fn[i] += 1;
                        // post j
                        if (jj > j-1){
                            FN_j[i] += 1;
                        }
                    }
                }
                if ( (*(y_valid + jj*numValid + i) == -1) & (*(y_pred + jj*numValid + i) == +1) ){
                    fp[i] += 1;
                    // post j
                    if (jj > j-1){
                        FP_j[i] += 1;
                    }
                }
            }else{
                // at index j
                if ( *(y_valid + jj*numValid + i) == +1 ){
                    tp[i] += 1;
                }else{
                    fp[i] += 1;
                }
            }
        }
        f[i] = F1_score(tp[i], fp[i], fn[i]);
        f_mean += f[i];
    }
    // mean F1-score
    f_mean /= (double)numValid;
    
    
    /************************************************
     * compute mean F1-score for varying thresholds
     ************************************************/
    for (i = 0; i < numValid; i++){
		// get the index of this instance
        // REMEMBER to minus 1!!!! Fxxk bug here!!!!
        ii = (mwSize)thresh_index[i] - 1;
        // update tp/fp/fn of instance at index ii
        if ( *(y_valid + (j-1)*numValid + ii) == -1 ){
            fp[ii] -= 1;
        }else{
            tp[ii] -= 1;
            fn[ii] += 1;
        }
        
        // update for labels j+1 ~ numLabel if necessary
        if ( (*(y_pred + (j-1)*numValid + ii) == +1) && (j < numLabel) ){
            // y_pred_new_ii[j-1] = -1;
            delta_y[j-1] = -1;
            for (jj = j; jj < numLabel; jj++){
                // no bias raw prediction
                // augment the features with previous labels 1 to j-1
                // remember to convert labels -1/+1 to 0/1 (necessary?)
                delta_wTx = 0;
                for (jjj=j-1; jjj<jj; jjj++){
					// delta_y = 0.5 * ( y_pred_new_ii[jjj] - y_pred_old_ii[jjj] );
                    delta_wTx += (delta_y[jjj] * (*(scut_w + jj*numValid + jjj)));
                }
                if ( *(wTx + jj*numValid + ii) + delta_wTx + scut_b[jj] > 0 ){
                    // y_pred_new_ii[jj] = +1;
					if ( *(y_valid + jj*numValid + ii) == +1 ){
						tp[ii] += 1;
					}else{
						fp[ii] += 1;
					}
                    delta_y[jj] = 0.5 * ( 1 - *(y_pred + jj*numValid + ii) );
                }else{
                    //y_pred_new_ii[jj] = -1;
                    if ( *(y_valid + jj*numValid + ii) == +1 ){
                        fn[ii] += 1;
                    }
                    delta_y[jj] = 0.5 * ( -1 - *(y_pred + jj*numValid + ii) );
                }
            }
            // update tp/fn/fp
            tp[ii] -= TP_j[ii];
            fn[ii] -= FN_j[ii];
            fp[ii] -= FP_j[ii];
        }
        // update F1-score of instance at index ii
        f_ii_new = F1_score(tp[ii], fp[ii], fn[ii]);
        f_mean += ((f_ii_new - f[ii]) / (double)numValid);
        F[i] = f_mean;
    }
    
    // free memory
    mxFree(delta_y);
	mxFree(tp);
	mxFree(fn);
	mxFree(fp);
	mxFree(TP_j);
	mxFree(FN_j);
	mxFree(FP_j);
	mxFree(f);
}


// the gateway function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    // initialize
    double *y_valid, *y_pred, *F;
    double *wTx, *scut_w, *scut_b;
    double *thresh_index;
    mwSize j;
    mwSize numValid, numLabel;
    mwSize mrows_pred, mrows_wTx, mrows_w, mrows_b, mrows_thresh, mrows_j;
    mwSize ncols_pred, ncols_wTx, ncols_w, ncols_b, ncols_thresh, ncols_j;
    
    // check parameters (we only check size, though you can also check type)
    numValid = (mwSize)mxGetM(prhs[0]);
    numLabel = (mwSize)mxGetN(prhs[0]);
    
    mrows_pred = (mwSize)mxGetM(prhs[1]);
    ncols_pred = (mwSize)mxGetN(prhs[1]);
    
    mrows_wTx = (mwSize)mxGetM(prhs[2]);
    ncols_wTx = (mwSize)mxGetN(prhs[2]);
    
    mrows_w = (mwSize)mxGetM(prhs[3]);
    ncols_w = (mwSize)mxGetN(prhs[3]);
    
    mrows_b = (mwSize)mxGetM(prhs[4]);
    ncols_b = (mwSize)mxGetN(prhs[4]);
    
    mrows_thresh = (mwSize)mxGetM(prhs[5]);
    ncols_thresh = (mwSize)mxGetN(prhs[5]);
    
    mrows_j = (mwSize)mxGetM(prhs[6]);
    ncols_j = (mwSize)mxGetN(prhs[6]);  
    
    if( (numValid != mrows_pred) || (numLabel != ncols_pred) ){
        mexErrMsgTxt("y_valid and y_pred should have the same dimension numValid x numLabel.");
    }
    if( (numValid != mrows_wTx) || (numLabel != ncols_wTx) ){
        mexErrMsgTxt("y_valid and wTx should have the same dimension numValid x numLabel.");
    }
    if( (mrows_w != numLabel) || (ncols_w != numLabel) ){
        mexErrMsgTxt("w should be a square array of size numLabel x numLabel.");
    }
    if( (mrows_b != numLabel && ncols_b == 1) || (mrows_b == 1 && ncols_b != numLabel) ){
        mexErrMsgTxt("b should be a vector of length numLabel.");
    }
    if( (numValid != mrows_thresh && ncols_thresh == 1) || (numValid != ncols_thresh && mrows_thresh == 1) ){
        mexErrMsgTxt("thresh_index should be a vector of length numValid.");
    }
    if( !(mrows_j==1 && ncols_j==1) ){
        mexErrMsgTxt("j should be a scalar index.");
    }
    
    // ground truth for validation data
    y_valid = mxGetPr(prhs[0]);    
    // prediction for validation data
    y_pred = mxGetPr(prhs[1]);
    // prediction for validation data without bias term
    wTx = mxGetPr(prhs[2]);
    // weights for all the binary classifiers
    scut_w = mxGetPr(prhs[3]);
    // bias for all the binary classifiers
    scut_b = mxGetPr(prhs[4]);    
    // index for threshold
    thresh_index = mxGetPr(prhs[5]);
    // index j
    j = (mwSize)mxGetScalar(prhs[6]);
    
    // set the output pointer to matrix F
    plhs[0] = mxCreateDoubleMatrix(numValid, 1, mxREAL);
    F = mxGetPr(plhs[0]);
    
    // call the computational routine
    cyclic_opt_update_F_CC_mex(y_valid, y_pred, wTx, scut_w, scut_b, thresh_index, j, F, numValid, numLabel);
  
}
