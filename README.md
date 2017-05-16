# HMM
Implementation of Hidden Markov Model (Forward &amp; Viterbi)  
cuteboydot@gmail.com

reference : http://www.kyobobook.co.kr/product/detailViewKor.laf?barcode=9788970859040  

> ## example : her life  
<br>
<img src="https://github.com/cuteboydot/HMM/blob/master/img/model.png" />
<img src="https://github.com/cuteboydot/HMM/blob/master/img/probmat.png" />
<img src="https://github.com/cuteboydot/HMM/blob/master/img/ABPI.png" />
</br>

> ## test1 : foward   
### Observation P(Walking, Walking, Cleaning, Shopping) = ???   
<br>
<img src="https://github.com/cuteboydot/HMM/blob/master/img/foward.png" />
<img src="https://github.com/cuteboydot/HMM/blob/master/img/fowardtest.png" />
</br>  
  
> ## test2 : viterbi  
### If observasion sequence = [Walking, Walking, Cleaning, Shopping]  
### Then state sequence = [?, ?, ?, ?]
<br>
<img src="https://github.com/cuteboydot/HMM/blob/master/img/viterbi.png" />
<img src="https://github.com/cuteboydot/HMM/blob/master/img/viterbitest.png" />
</br>

- code  
```python
import numpy as np

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PARAMETER SETTING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#################################################################################
# STATE 0 = RAINNY, STATE 1 = SUNNY
# OBSERVATION 0 = WALKING, OBSERVATION 1 = SHOPPING, OBSERVATON 2 = CLEANING
#################################################################################
SIZE_STATE = 2
SIZE_OBSERVATION = 3

#################################################################################
# TRANSITION PROBABILITY OF STATE
# P(RAINNY | RAINNY) = 0.7, P(SUNNY | RAINNY) = 0.3
# P(RAINNY | SUNNY)  = 0.4, P(SUNNY | SUNNY) = 0.6
#################################################################################
A = np.zeros(shape=(SIZE_STATE, SIZE_STATE), dtype=np.float)
A[0][0] = 0.7
A[0][1] = 0.3
A[1][0] = 0.4
A[1][1] = 0.6

#################################################################################
# OBSERVATION PROBABILITY
# P(WALKING | RAINNY) = 0.1, P(OSHOPPING1 | RAINNY) = 0.4, P(CLEANING | RAINNY) = 0.5
# P(WALKING | SUNNY)  = 0.6, P(SHOPPING | SUNNY)    = 0.3, P(CLEANING | SUNNY)  = 0.1
#################################################################################
B = np.zeros(shape=(SIZE_STATE, SIZE_OBSERVATION))
B[0][0] = 0.1
B[0][1] = 0.4
B[0][2] = 0.5
B[1][0] = 0.6
B[1][1] = 0.3
B[1][2] = 0.1

#################################################################################
# INITIAL TRANSITION PROBABILITY
# START AS P(S0) = 0.6, P(S1) = 0.4
#################################################################################
PI = np.zeros(shape=(SIZE_STATE), dtype=np.float)
PI[0] = 0.6
PI[1] = 0.4
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
END PARAMETER SETTING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def forward (ob_seq) :
    prob_sum = 0
    trans_mat = np.zeros(shape=(len(ob_seq), SIZE_STATE), dtype=np.float)

    for a in range(len(ob_seq)) :
        for b in range(SIZE_STATE) :
            if (a == 0) :
                trans_mat[a][b] = PI[b] * B[b][ob_seq[a]]
                print ('SEQ %d : %.1f * %.1f = %.4f' % (a, PI[b], B[b][ob_seq[a]], trans_mat[a][b]))
                continue

            prob_cur = 0
            print ('SEQ %d :(' % (a), end='')
            for c in range(SIZE_STATE) :
                prob_cur += trans_mat[a-1][c] * A[c][b]
                print ('%.4f * %.1f' % (trans_mat[a-1][c], A[c][b]), end='')
                if(c < SIZE_STATE-1):
                    print (' + ', end='')

            trans_mat[a][b] = prob_cur * B[b][ob_seq[a]]
            print (') * %.1f = %.4f' % (B[b][ob_seq[a]], trans_mat[a][b]))

        prob_sum = np.sum(trans_mat[a])
    return prob_sum, trans_mat


def viterbi (ob_seq) :
    st_seq = np.zeros(shape=(len(ob_seq)), dtype=np.int)
    trans_mat = np.zeros(shape=(len(ob_seq), SIZE_STATE), dtype=np.float)

    for a in range(len(ob_seq)) :
        for b in range(SIZE_STATE) :
            if (a == 0) :
                trans_mat[a][b] = PI[b] * B[b][ob_seq[a]]
                print ('SEQ %d : %.1f * %.1f = %.4f' % (a, PI[b], B[b][ob_seq[a]], trans_mat[a][b]))
            else :
                pre_max = np.argmax(trans_mat[a-1])
                trans_mat[a][b] = trans_mat[a-1][pre_max] * A[pre_max][b] * B[b][ob_seq[a]]
                print ('SEQ %d : %.4f * %.1f * %.1f = %.4f' % (a, trans_mat[a-1][pre_max], A[pre_max][b], B[b][ob_seq[a]], trans_mat[a][b]))

        state_max = np.argmax(trans_mat[a])
        st_seq[a] = state_max
    return st_seq, trans_mat


print ()
print ('A : TRANSITION PROBABILITY OF STATE')
print (A)
print ('B : OBSERVATION PROBABILITY')
print (B)
print ('PI : INITIAL TRANSITION PROBABILITY')
print (PI)

observation_seq = np.zeros(shape=(4), dtype=np.int)
observation_seq[0] = 0
observation_seq[1] = 0
observation_seq[2] = 2
observation_seq[3] = 1

print ()
print ('FOWARD TEST')
print ('QUERY => P(%s) ??' % (str(observation_seq)))
print ()

(prob_sum, trans_mat) = forward(observation_seq)
print ('Transition matrix')
print (trans_mat)
print ()
print ('RESULT => %.6f' % (prob_sum))

print ()
print ('VIERTBI TEST')
print ('QUERY => %s' % (str(observation_seq)))
print ()

(sate_seq, trans_mat) = viterbi(observation_seq)

print ('Transition matrix')
print (trans_mat)
print ()

print ('EXPECTED STATE SEQUENCE')
print ('RESULT => %s' % (str(sate_seq)))
```
